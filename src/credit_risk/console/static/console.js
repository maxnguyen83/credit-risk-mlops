/* Credit-default console.
 *
 * Every value that comes from Airflow or from a results file is put on the
 * page as text (textContent / text nodes), never parsed as markup.
 * Polling: every 3 s while a run is queued or running, every 15 s otherwise.
 */
(function () {
  "use strict";

  const FAST_POLL_MS = 3000;
  const SLOW_POLL_MS = 15000;
  const ACTIVE = new Set(["queued", "running"]);

  const STATES = {
    success: ["Thành công", "s-ok", "✓"],
    running: ["Đang chạy", "s-run", "●"],
    queued: ["Đang chờ", "s-wait", "…"],
    scheduled: ["Đã lên lịch", "s-wait", "…"],
    deferred: ["Tạm hoãn", "s-wait", "…"],
    up_for_retry: ["Chờ thử lại", "s-warn", "↻"],
    up_for_reschedule: ["Chờ lên lịch lại", "s-warn", "↻"],
    restarting: ["Đang khởi động lại", "s-warn", "↻"],
    failed: ["Thất bại", "s-bad", "✕"],
    upstream_failed: ["Lỗi ở bước trước", "s-bad", "✕"],
    skipped: ["Bỏ qua", "s-idle", "–"],
    removed: ["Đã gỡ", "s-idle", "–"],
  };
  const NOT_RUN = ["Chưa chạy", "s-idle", "○"];

  const RISK = {
    high: ["Cao", "s-bad"],
    medium: ["Trung bình", "s-warn"],
    low: ["Thấp", "s-ok"],
  };

  const state = {
    timer: null,
    poll: 0,
    scoringKey: null,
    maxBytes: 5 * 1024 * 1024,
    pipelines: {},
    available: false,
  };

  // ------------------------------------------------------------ helpers

  const $ = (id) => document.getElementById(id);

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else node.setAttribute(key, value === true ? "" : String(value));
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  const intFmt = new Intl.NumberFormat("vi-VN");

  function fmtInt(value) {
    return typeof value === "number" && isFinite(value) ? intFmt.format(value) : "—";
  }

  function fmtPercent(fraction, digits) {
    if (typeof fraction !== "number" || !isFinite(fraction)) return "—";
    return (fraction * 100).toLocaleString("vi-VN", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    }) + " %";
  }

  function fmtTime(iso) {
    if (!iso) return "—";
    const date = new Date(iso);
    if (isNaN(date.getTime())) return String(iso);
    return date.toLocaleString("vi-VN", {
      hour: "2-digit", minute: "2-digit", second: "2-digit",
      day: "2-digit", month: "2-digit", year: "numeric",
    });
  }

  function fmtClock(iso) {
    const date = new Date(iso || "");
    return isNaN(date.getTime()) ? "—" : date.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
  }

  function fmtDate(iso) {
    const date = new Date(iso || "");
    return isNaN(date.getTime()) ? "" : date.toLocaleDateString("vi-VN", { day: "2-digit", month: "2-digit", year: "numeric" });
  }

  function fmtDuration(seconds) {
    if (typeof seconds !== "number" || !isFinite(seconds) || seconds < 0) return "";
    if (seconds < 1) return "< 1 giây";
    if (seconds < 60) return Math.round(seconds) + " giây";
    const minutes = Math.floor(seconds / 60);
    const rest = Math.round(seconds % 60);
    if (minutes < 60) return minutes + " phút" + (rest ? " " + rest + " giây" : "");
    return Math.floor(minutes / 60) + " giờ " + (minutes % 60) + " phút";
  }

  function elapsed(start, end) {
    const from = Date.parse(start);
    const to = end ? Date.parse(end) : Date.now();
    return isFinite(from) && isFinite(to) ? Math.max(0, (to - from) / 1000) : null;
  }

  // Task ids wrap at their underscores rather than mid-word in a narrow box.
  function breakable(text) {
    const parts = String(text).split("_");
    return parts.flatMap((part, index) => (index < parts.length - 1 ? [part + "_", el("wbr")] : [part]));
  }

  function describe(stateName) {
    return STATES[stateName] || NOT_RUN;
  }

  async function api(path, options) {
    const init = Object.assign({ headers: { Accept: "application/json" } }, options || {});
    let response;
    try {
      response = await fetch(path, init);
    } catch (error) {
      return { ok: false, status: 0, body: { message: "Không kết nối được bảng điều khiển. Kiểm tra mạng rồi thử lại." } };
    }
    let body = null;
    try {
      body = await response.json();
    } catch (error) {
      body = null;
    }
    return { ok: response.ok, status: response.status, body: body || {} };
  }

  function flash(id, kind, message, items) {
    const node = $(id);
    node.className = "flash" + (kind ? " flash-" + kind : "");
    node.replaceChildren(message);
    if (items && items.length) {
      node.append(el("ul", null, items.map((item) => el("li", null, item))));
    }
    node.hidden = false;
  }

  function errorMessage(result) {
    const body = result.body || {};
    return body.message || "Yêu cầu không thành công (HTTP " + result.status + ").";
  }

  // ---------------------------------------------------------- pipelines

  function renderFlow(listId, pipeline) {
    const items = (pipeline.tasks || []).map((task, index) => {
      const [label, cls, icon] = describe(task.state);
      let timing = "";
      if (task.state === "running") timing = fmtDuration(elapsed(task.start_date, null));
      else if (typeof task.duration === "number") timing = fmtDuration(task.duration);
      return el(
        "li",
        { class: "task " + cls, title: task.task_id + " — " + label },
        el("span", { class: "task-index" }, "Bước " + (index + 1)),
        el("span", { class: "task-label" }, task.label || task.task_id),
        el("span", { class: "task-id" }, breakable(task.task_id)),
        el("span", { class: "task-state" }, icon + " " + label + (timing ? " · " + timing : ""))
      );
    });
    $(listId).replaceChildren(...items);
  }

  function renderMeta(metaId, pipeline, available) {
    const parts = [];
    if (!available) {
      parts.push(el("span", null, "Không lấy được trạng thái từ Airflow — sơ đồ hiển thị các bước theo thiết kế."));
    } else if (!pipeline.found) {
      parts.push(el("span", { class: "badge s-bad" }, "Chưa nạp"));
      parts.push(el("span", null, pipeline.reason || "Airflow chưa nạp DAG này."));
    } else if (!pipeline.latest_run) {
      parts.push(el("span", null, "Chưa có lần chạy nào."));
    } else {
      const run = pipeline.latest_run;
      const [label, cls] = describe(run.state);
      const duration = fmtDuration(elapsed(run.start_date, ACTIVE.has(run.state) ? null : run.end_date));
      parts.push(el("span", null, "Lần chạy gần nhất ", el("strong", null, fmtTime(run.start_date || run.logical_date))));
      parts.push(el("span", { class: "badge " + cls }, label));
      if (duration) parts.push(el("span", null, "Thời gian ", el("strong", null, duration)));
      if (run.input) parts.push(el("span", null, "Đầu vào ", el("code", null, run.input)));
      if (pipeline.n_active_runs > 1) parts.push(el("span", null, "+" + (pipeline.n_active_runs - 1) + " lần chạy đang chờ"));
    }
    if (available && pipeline.found && pipeline.is_paused) {
      parts.push(el("span", { class: "badge" }, "DAG đang tạm dừng — bấm chạy sẽ tự bật lại"));
    }
    if (pipeline.airflow_link) {
      parts.push(el("a", { href: pipeline.airflow_link, target: "_blank", rel: "noopener" }, "Xem nhật ký trong Airflow ↗"));
    }
    $(metaId).replaceChildren(...parts);
  }

  function legend() {
    return el(
      "div",
      { class: "legend", "aria-hidden": "true" },
      el("span", { class: "s-ok" }, "Thành công"),
      el("span", { class: "s-run" }, "Đang chạy"),
      el("span", { class: "s-wait" }, "Đang chờ"),
      el("span", { class: "s-bad" }, "Thất bại"),
      el("span", null, "Chưa chạy / bỏ qua")
    );
  }

  function setDisabled(node, disabled, reason) {
    if (node.tagName === "BUTTON") node.disabled = disabled;
    else node.setAttribute("aria-disabled", disabled ? "true" : "false");
    node.title = disabled ? reason : "";
  }

  function blockedReason(pipeline, available) {
    if (!available) return "Airflow chưa kết nối được.";
    if (!pipeline || !pipeline.found) return "Airflow chưa nạp pipeline này.";
    if (pipeline.active) return "Pipeline đang chạy, đợi lần chạy hiện tại xong.";
    return "";
  }

  function renderControls() {
    const training = state.pipelines.training;
    const scoring = state.pipelines.scoring;
    const trainingBlocked = blockedReason(training, state.available);
    const scoringBlocked = blockedReason(scoring, state.available);
    setDisabled($("train-button"), Boolean(trainingBlocked), trainingBlocked);
    setDisabled($("train-confirm-yes"), Boolean(trainingBlocked), trainingBlocked);
    setDisabled($("sample-button"), Boolean(scoringBlocked), scoringBlocked);
    setDisabled($("upload-label"), Boolean(scoringBlocked), scoringBlocked);
    $("upload-input").disabled = Boolean(scoringBlocked);
  }

  function renderAirflowStatus(view) {
    const chip = $("airflow-chip");
    chip.className = "chip " + (view.available ? "chip-ok" : "chip-bad");
    $("airflow-chip-text").textContent = view.available ? "Airflow: đã kết nối" : "Airflow: mất kết nối";
    const banner = $("airflow-banner");
    banner.hidden = view.available;
    if (!view.available) {
      banner.textContent = (view.reason || "Không kết nối được Airflow.") +
        " Trang vẫn hiển thị kết quả đã có; các nút chạy pipeline tạm khoá.";
    }
    $("updated-at").textContent = "Cập nhật lúc " + new Date().toLocaleTimeString("vi-VN");
  }

  // Called by the timer, after every action and when the tab comes back into
  // view. Only the newest call renders and re-arms the timer, so overlapping
  // calls can neither paint stale state nor start a second polling loop.
  async function refreshPipelines() {
    const poll = ++state.poll;
    clearTimeout(state.timer);
    const result = await api("/api/pipelines");
    if (poll !== state.poll) return;
    let anyActive = false;
    if (result.ok && Array.isArray(result.body.pipelines)) {
      const view = result.body;
      state.available = Boolean(view.available);
      renderAirflowStatus(view);
      for (const pipeline of view.pipelines) {
        state.pipelines[pipeline.role] = pipeline;
        renderFlow(pipeline.role + "-flow", pipeline);
        renderMeta(pipeline.role + "-meta", pipeline, state.available);
        anyActive = anyActive || Boolean(pipeline.active);
      }
      renderControls();
      watchScoring(state.pipelines.scoring);
    } else {
      state.available = false;
      renderAirflowStatus({ available: false, reason: errorMessage(result) });
      renderControls();
    }
    state.timer = setTimeout(refreshPipelines, anyActive ? FAST_POLL_MS : SLOW_POLL_MS);
  }

  // A scoring run that reaches a final state has just published (or failed
  // to publish) a call list, so section 3 is re-read exactly then.
  function watchScoring(scoring) {
    const run = scoring && scoring.latest_run;
    const key = run ? run.run_id + "|" + run.state : "";
    if (state.scoringKey !== null && key !== state.scoringKey && run && !ACTIVE.has(run.state)) {
      refreshResults();
    }
    state.scoringKey = key;
  }

  async function startTraining() {
    $("train-confirm").hidden = true;
    setDisabled($("train-button"), true, "Đang gửi yêu cầu…");
    const result = await api("/api/pipelines/credit_risk_pipeline/runs", { method: "POST" });
    if (result.ok) {
      flash("training-flash", "ok", "Đã bắt đầu huấn luyện — mã lần chạy " + result.body.run_id + "." +
        (result.body.unpaused ? " DAG đã được bật lại." : ""));
    } else {
      flash("training-flash", "bad", errorMessage(result));
    }
    refreshPipelines();
  }

  async function startSample() {
    setDisabled($("sample-button"), true, "Đang gửi yêu cầu…");
    const result = await api("/api/batches/sample", { method: "POST" });
    if (result.ok) {
      flash("scoring-flash", "ok", "Đang chấm điểm danh sách mẫu 5.000 khách — mã lần chạy " + result.body.run_id + ".");
    } else {
      flash("scoring-flash", "bad", errorMessage(result));
    }
    refreshPipelines();
  }

  async function uploadFile() {
    const input = $("upload-input");
    const file = input.files && input.files[0];
    input.value = "";
    if (!file) return;
    if (file.size > state.maxBytes) {
      flash("scoring-flash", "bad", "File " + file.name + " lớn hơn giới hạn " + fmtMegabytes(state.maxBytes) + ".");
      return;
    }
    flash("scoring-flash", null, "Đang tải lên " + file.name + "…");
    setDisabled($("upload-label"), true, "Đang tải lên…");
    const result = await api("/api/batches", {
      method: "POST",
      headers: { "Content-Type": "text/csv", Accept: "application/json" },
      body: file,
    });
    if (result.ok) {
      flash("scoring-flash", "ok", "Đã nhận " + file.name + " (" + fmtInt(result.body.n_rows) +
        " khách). Pipeline chấm điểm đã tự chạy — mã lần chạy " + result.body.run_id + ".");
    } else {
      flash("scoring-flash", "bad", errorMessage(result));
    }
    refreshPipelines();
  }

  function fmtMegabytes(bytes) {
    return (bytes / (1024 * 1024)).toLocaleString("vi-VN", { maximumFractionDigits: 1 }) + " MB";
  }

  // ------------------------------------------------------------ results

  function kpi(label, value, sub, accent) {
    return el(
      "div",
      { class: "kpi" + (accent ? " kpi-accent" : "") },
      el("div", { class: "kpi-label" }, label),
      el("div", { class: "kpi-value" }, value),
      sub ? el("div", { class: "kpi-sub" }, sub) : null
    );
  }

  function renderKpis(summary) {
    const capacity = typeof summary.capacity_fraction === "number"
      ? "năng lực liên hệ: " + fmtPercent(summary.capacity_fraction, 0) + " số khách đã chấm"
      : "theo năng lực liên hệ của bộ phận thu hồi";
    const rejected = typeof summary.n_rejected === "number" && summary.n_rejected > 0
      ? fmtInt(summary.n_rejected) + " dòng bị loại khi kiểm tra"
      : "Không có dòng bị loại";
    const threshold = typeof summary.threshold_used === "number"
      ? summary.threshold_used.toLocaleString("vi-VN", { maximumFractionDigits: 4 })
      : "—";
    const version = summary.model_version !== null && summary.model_version !== undefined
      ? "v" + summary.model_version
      : "—";
    $("kpis").replaceChildren(
      kpi("Số khách đã chấm", fmtInt(summary.n_scored), rejected),
      kpi("Số khách cần gọi (top 10%)", fmtInt(summary.n_call_list), capacity, true),
      kpi("Số vượt ngưỡng", fmtInt(summary.n_above_threshold), "xác suất ≥ ngưỡng quyết định"),
      kpi("Model version", version, summary.model_name || ""),
      kpi("Ngưỡng", threshold, "ngưỡng quyết định của model"),
      kpi("Thời điểm", fmtClock(summary.generated_at), [fmtDate(summary.generated_at), summary.input_path ? "nguồn: " + summary.input_path : ""].filter(Boolean).join(" · "))
    );
  }

  function probabilityCell(probability) {
    const bar = el("span");
    if (typeof probability === "number") bar.style.width = Math.max(0, Math.min(100, probability * 100)) + "%";
    return el("div", { class: "prob" }, fmtPercent(probability, 1), el("div", { class: "prob-bar" }, bar));
  }

  function riskBadge(band) {
    const [label, cls] = RISK[band] || [band || "—", ""];
    return el("span", { class: "badge " + cls }, label);
  }

  function renderRows(rows) {
    const body = $("call-rows");
    if (!rows.length) {
      body.replaceChildren(el("tr", null, el("td", { colspan: 5, class: "muted" }, "Danh sách gọi trống hoặc chưa đọc được.")));
      return;
    }
    body.replaceChildren(...rows.map((row) => el(
      "tr",
      null,
      el("td", { class: "num" }, row.rank === null ? "—" : fmtInt(row.rank)),
      el("td", { class: "account" }, row.account_id || "—"),
      el("td", null, probabilityCell(row.default_probability)),
      el("td", null, riskBadge(row.risk_band)),
      el("td", null, row.top_reasons && row.top_reasons.length
        ? el("ul", { class: "reasons" }, row.top_reasons.map((reason) => el("li", null, reason)))
        : el("span", { class: "muted" }, "—"))
    )));
  }

  function setDownload(id, url) {
    const link = $(id);
    setDisabled(link, !url, "Lần chấm gần nhất không có file này.");
    if (url) link.setAttribute("href", url);
  }

  function showEmpty(reason) {
    $("results-body").hidden = true;
    $("results-empty").hidden = false;
    $("results-empty-reason").textContent = reason;
    setDownload("download-call-list", null);
    setDownload("download-scores", null);
  }

  async function refreshResults() {
    const result = await api("/api/results/latest");
    if (!result.ok || result.body.available === false) {
      showEmpty(result.body.reason || errorMessage(result));
      return;
    }
    const data = result.body;
    const summary = data.summary || {};
    const rows = Array.isArray(data.rows) ? data.rows : [];
    $("results-empty").hidden = true;
    $("results-body").hidden = false;
    renderKpis(summary);
    renderRows(rows);
    const downloads = data.downloads || {};
    setDownload("download-call-list", downloads["call_list.csv"]);
    setDownload("download-scores", downloads["scores.csv"]);
    const total = typeof data.n_rows_total === "number" ? data.n_rows_total : rows.length;
    $("table-caption").textContent = rows.length
      ? "Hiển thị " + fmtInt(rows.length) + " / " + fmtInt(total) + " khách đầu danh sách gọi. Tải call_list.csv để có đầy đủ." +
        (summary.dag_run_id ? " Lần chạy: " + summary.dag_run_id + "." : "")
      : "";
  }

  // -------------------------------------------------------- static data

  async function loadLinks() {
    const result = await api("/api/links");
    const links = result.ok && Array.isArray(result.body.links) ? result.body.links : [];
    $("link-grid").replaceChildren(...links.map((link) => el(
      "a",
      { class: "link-card", href: link.url, target: "_blank", rel: "noopener" },
      el("span", { class: "link-title" }, link.title),
      el("span", { class: "link-desc" }, link.description),
      el("span", { class: "link-url" }, link.url)
    )));
  }

  async function loadRequirements() {
    const result = await api("/api/batches/requirements");
    if (!result.ok) return;
    if (Array.isArray(result.body.required_columns)) {
      $("required-columns").textContent = result.body.required_columns.join(", ");
    }
    if (typeof result.body.max_bytes === "number") {
      state.maxBytes = result.body.max_bytes;
      $("max-size").textContent = fmtMegabytes(state.maxBytes);
    }
  }

  // ------------------------------------------------------------- wiring

  function init() {
    for (const id of ["training-flow", "scoring-flow"]) $(id).after(legend());
    $("train-button").addEventListener("click", () => {
      $("train-confirm").hidden = false;
      $("train-confirm-yes").focus();
    });
    $("train-confirm-no").addEventListener("click", () => {
      $("train-confirm").hidden = true;
      $("train-button").focus();
    });
    $("train-confirm-yes").addEventListener("click", startTraining);
    $("sample-button").addEventListener("click", startSample);
    $("upload-input").addEventListener("change", uploadFile);
    $("upload-label").addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        $("upload-input").click();
      }
    });
    $("upload-label").setAttribute("tabindex", "0");
    $("upload-label").setAttribute("role", "button");
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) refreshPipelines();
    });

    loadLinks();
    loadRequirements();
    refreshResults();
    refreshPipelines();
  }

  init();
})();

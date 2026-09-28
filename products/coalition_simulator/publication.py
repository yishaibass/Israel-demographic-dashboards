from __future__ import annotations


DEMOGRAPHICS_TAB = '  <button class="tab-btn" data-tab="demographics">ניתוח דמוגרפי</button>'

DEMOGRAPHICS_PANEL = """<section class="tab-panel" id="tab-demographics">
  <iframe id="demographicsFrame" class="demographics-frame" src="demographics.html?embed=1" title="ניתוח דפוסי הצבעה לפי דמוגרפיה ומפלגה"></iframe>
</section>"""

HOME_BUTTON = """<style>.hb-home-back{position:fixed;top:14px;left:14px;z-index:99999;background:#0f172a;color:#fff;padding:9px 16px;border-radius:999px;text-decoration:none;font-family:'Heebo',system-ui,-apple-system,sans-serif;font-size:14px;font-weight:600;box-shadow:0 4px 14px rgba(0,0,0,.28);display:inline-flex;align-items:center;gap:6px}.hb-home-back .hb-lbl{display:inline}@media(max-width:480px){.hb-home-back{padding:0;width:40px;height:40px;gap:0;justify-content:center;border-radius:50%;font-size:18px}.hb-home-back .hb-lbl{display:none}}</style>
<a href="../index.html" class="hb-home-back" aria-label="חזרה לדף הבית">🏠<span class="hb-lbl">דף הבית</span></a>"""

IFRAME_RESIZER = """<script>
(function () {
  var frame = document.getElementById('demographicsFrame');
  if (!frame) return;
  function resizeDemographicsFrame() {
    try {
      var doc = frame.contentDocument;
      if (!doc) return;
      var height = Math.max(doc.documentElement.scrollHeight, doc.body ? doc.body.scrollHeight : 0);
      if (height) frame.style.height = height + 'px';
    } catch (e) {}
  }
  frame.addEventListener('load', function () {
    resizeDemographicsFrame();
    try { new ResizeObserver(resizeDemographicsFrame).observe(frame.contentDocument.body); } catch (e) {}
  });
  window.addEventListener('resize', resizeDemographicsFrame);
  document.querySelector('[data-tab="demographics"]').addEventListener('click', function () {
    setTimeout(resizeDemographicsFrame, 80);
  });
})();
</script>"""


def _replace_once(html: str, old: str, new: str, label: str) -> str:
    count = html.count(old)
    if count != 1:
        raise ValueError(f"Coalition publication anchor {label!r} occurred {count} times")
    return html.replace(old, new, 1)


def _model_payload(html: str) -> str:
    start = html.index("const DATA = ")
    end = html.index("\n};", start) + len("\n};")
    return html[start:end]


def integrate_publication_shell(html: str) -> str:
    """Restore the publication shell recovered from remote commit f0423cf.

    The legacy builder remains authoritative for the validated coalition DATA
    payload. This transform owns only cross-product navigation and embedding.
    """
    model_before = _model_payload(html)

    if "button.tab-btn, a.tab-btn" not in html:
        html = _replace_once(html, "button.tab-btn {", "button.tab-btn, a.tab-btn {", "tab link base style")
        html = _replace_once(
            html,
            "margin-bottom: -1px;\n  }",
            "margin-bottom: -1px; text-decoration: none; display: inline-block;\n  }",
            "tab link layout",
        )
        html = _replace_once(
            html,
            "button.tab-btn:hover { color: var(--text-primary); }",
            "button.tab-btn:hover, a.tab-btn:hover { color: var(--text-primary); }",
            "tab link hover style",
        )
        html = _replace_once(
            html,
            "button.tab-btn.active { color: var(--coalition); border-bottom-color: var(--coalition); }",
            "button.tab-btn.active, a.tab-btn.active { color: var(--coalition); border-bottom-color: var(--coalition); }",
            "tab link active style",
        )

    if 'data-tab="demographics"' not in html:
        html = _replace_once(
            html,
            '  <button class="tab-btn" data-tab="parties">שיוך מפלגות</button>',
            '  <button class="tab-btn" data-tab="parties">שיוך מפלגות</button>\n' + DEMOGRAPHICS_TAB,
            "demographics tab",
        )
    if 'id="tab-demographics"' not in html:
        html = _replace_once(
            html,
            '<footer class="foot">',
            DEMOGRAPHICS_PANEL + '\n<footer class="foot">',
            "demographics panel",
        )
    if ".demographics-frame" not in html:
        html = _replace_once(
            html,
            "</style>\n</head>",
            "  .demographics-frame { width: 100%; border: 0; display: block; min-height: 1500px; background: var(--page); }</style>\n</head>",
            "demographics style",
        )
    if 'class="hb-home-back"' not in html:
        html = _replace_once(
            html,
            "\n</body>",
            HOME_BUTTON + "\n\n" + IFRAME_RESIZER + "</body>",
            "publication navigation",
        )

    if _model_payload(html) != model_before:
        raise ValueError("Publication integration changed the coalition model payload")
    verify_publication_shell(html)
    return html


def verify_publication_shell(html: str) -> None:
    required = {
        "demographic tab": 'data-tab="demographics"',
        "demographic panel": 'id="tab-demographics"',
        "demographic iframe": 'src="demographics.html?embed=1"',
        "home navigation": 'href="../index.html" class="hb-home-back"',
        "iframe resize hook": "new ResizeObserver(resizeDemographicsFrame)",
    }
    missing = [label for label, marker in required.items() if marker not in html]
    if missing:
        raise ValueError("Coalition publication shell is incomplete: " + ", ".join(missing))

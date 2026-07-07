"""键盘快捷键 HTML 组件。"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


HOTKEY_JS = """
<script>
(function() {
  const doc = window.parent.document;
  if (doc._cellInspectorHotkeys) return;
  doc._cellInspectorHotkeys = true;

  doc.addEventListener('keydown', function(e) {
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || e.target.isContentEditable) return;

    const map = {
      'f': 'F', 'F': 'F',
      'b': 'B', 'B': 'B',
      's': 'S', 'S': 'S',
      'u': 'U', 'U': 'U',
      'r': 'R', 'R': 'R',
      'ArrowLeft': 'LEFT',
      'ArrowRight': 'RIGHT'
    };
    const key = map[e.key];
    if (!key) return;
    e.preventDefault();

    const inputs = doc.querySelectorAll('input[data-testid="stTextInput"] input');
    let target = null;
    for (const inp of inputs) {
      if (inp.placeholder === '__hotkey__') {
        target = inp;
        break;
      }
    }
    if (target) {
      target.value = key + '_' + Date.now();
      target.dispatchEvent(new Event('input', { bubbles: true }));
    }
  });
})();
</script>
"""


def render_hotkey_listener() -> str | None:
    """注入快捷键监听，返回捕获到的按键。"""
    components.html(HOTKEY_JS, height=0)
    st.markdown(
        """
        <style>
        div[data-testid="stTextInput"]:has(input[placeholder="__hotkey__"]) {
            position: absolute; left: -9999px; height: 0; overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    val = st.text_input(
        "快捷键",
        value="",
        key="_hotkey_capture",
        label_visibility="collapsed",
        placeholder="__hotkey__",
    )
    if val and "_" in val:
        return val.rsplit("_", 1)[0]
    return None

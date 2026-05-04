# -*- coding: utf-8 -*-
"""
分组图标 SVG 字符串集合
用法：from assets.icons import GROUP_ICONS
所有图标统一 24x24 viewBox，stroke=currentColor 便于跟随主题色
"""

# 基本信息（用户/身份图标）
ICON_BASIC = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>"""

# 血常规（血滴图标）
ICON_BLOOD = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.5s5.5 6.5 5.5 11a5.5 5.5 0 0 1-11 0c0-4.5 5.5-11 5.5-11Z"/><circle cx="10" cy="13" r="1" fill="currentColor"/></svg>"""

# 炎症（火焰/温度计图标）
ICON_INFLAM = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2c1 6 6 7 6 12a6 6 0 0 1-12 0c0-3 2-4 3-7 0 0 1 2 3 2 0-2-1-3 0-7Z"/></svg>"""

# 肝功（化学瓶/烧瓶图标）
ICON_LIVER = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3h6v5l4 9a3 3 0 0 1-3 4H8a3 3 0 0 1-3-4l4-9V3Z"/><line x1="9" y1="3" x2="15" y2="3"/></svg>"""

# 肾功（圆形器官 + 滤过纹路）
ICON_KIDNEY = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 4c4 0 6 3 6 8s-2 8-6 8c-3 0-4-2-4-4 0-3 1-4 1-6S4 4 7 4Z"/><path d="M17 4c-4 0-6 3-6 8s2 8 6 8c3 0 4-2 4-4 0-3-1-4-1-6s1-6-3-6Z"/></svg>"""

# 心肌酶（心脏图标）
ICON_HEART = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>"""

GROUP_ICONS = {
    "basic": ICON_BASIC,
    "blood": ICON_BLOOD,
    "inflam": ICON_INFLAM,
    "liver": ICON_LIVER,
    "kidney": ICON_KIDNEY,
    "heart": ICON_HEART,
}

# Tab 图标
ICON_NOMOGRAM = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/><circle cx="9" cy="6" r="2" fill="currentColor"/><circle cx="15" cy="12" r="2" fill="currentColor"/><circle cx="11" cy="18" r="2" fill="currentColor"/></svg>"""

ICON_SHAP = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l3-9 4 18 3-9h4"/></svg>"""

ICON_COMPARE = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="18" rx="1"/><rect x="14" y="3" width="7" height="18" rx="1"/></svg>"""

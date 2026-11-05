# -*- coding: utf-8 -*-
"""result.json + 라이브러리를 단일 HTML(H23_물동량_대시보드.html)로 결합"""
import json
t = open("dashboard_template.html", encoding="utf-8").read()
data = open("data/result.json", encoding="utf-8").read()
t = t.replace("/*__CHARTJS__*/", open("lib/chart.min.js", encoding="utf-8").read().replace("</script>", r"<\/script>"))
t = t.replace("/*__XLSX__*/", open("lib/xlsx.min.js", encoding="utf-8").read().replace("</script>", r"<\/script>"))
t = t.replace("/*__DATA__*/", data)
open("H23_물동량_대시보드.html", "w", encoding="utf-8").write(t)
print("built", len(t) // 1024, "KB")

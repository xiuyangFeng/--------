# 派生字段审计更新

旧derived_audit实现与结果已作废，不能把旧error_count解释为病例失败，也不能据它自动放行。现由 `wss_v5/audit_refined_geometry.py`独立核验全部170例：面积/周长/质心、多边形、非均匀s坡度、上下游10mm上下文、壁面映射与逐字段mask均通过。见 [最终结果](../../算法优化_20260915/README.md)。

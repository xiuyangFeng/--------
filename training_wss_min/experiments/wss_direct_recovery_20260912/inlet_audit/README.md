# Frozen geometry and E9 input audit

172/172 cases passed.

- Public peak Q: 0.000107269066557 m³/s (step 1162; bit-identical for all cases).
- Inlet area range: 231.450–999.260 mm², derived from actual inlet face geometry.
- Q/A range: 0.107349–0.463466 m/s.
- Local normal/tangent/radius/scale metadata passed frozen node identity, raw coordinate and finiteness checks.

This feature describes the geometry-dependent reference speed under a common prescribed waveform. It adds no patient-specific flow or outlet information. Only the existing frozen geometry and the public nominal waveform are used; no CFD solution, measured flux, RCR, or replacement geometry is read. The source inlet boundary must be available at deployment, or its surface must be triangulated to obtain the same area definition.

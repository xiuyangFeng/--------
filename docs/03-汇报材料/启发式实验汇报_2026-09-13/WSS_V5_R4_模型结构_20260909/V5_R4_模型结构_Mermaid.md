# V5·R4 模型结构：Mermaid 精简版

突出主路径通道变化：`17 → 32 → 64 → 128 → 256 → 128 → 64 → 32 → 64 → 1`。

## 01_整体网络结构.mmd

```mermaid
flowchart LR
    I["输入<br/>5000×17"] --> S["Stem<br/>5000×32"]
    S --> E1["SA1<br/>125×64"] --> E2["SA2 + L-SA2<br/>125×128"] --> E3["SA3<br/>32×256"]
    E3 --> D3["FP3 + skip<br/>125×128"] --> D2["FP2 + skip<br/>125×64"] --> D1["FP1 + skip<br/>5000×32"]
    D1 --> Q["Query 特征<br/>Nq×32"] --> H["回归头<br/>32→64→1"] --> O["WSS(Pa)<br/>Nq×1"]
    S -. "32D skip" .-> D1
    E1 -. "64D skip" .-> D2
    E2 -. "128D skip" .-> D3
    E2 -. "SA2 内部：LocalGeoPE → 4头局部Transformer → max → InvRes" .-> NOTE["核心模块说明"]
    classDef input fill:#E8F1FB,stroke:#2878BD,color:#12304A;
    classDef enc fill:#DCECFB,stroke:#2878BD,color:#12304A;
    classDef attn fill:#F0E5FA,stroke:#8650B5,color:#3C2055;
    classDef dec fill:#E1F3EF,stroke:#0B8F88,color:#123F3B;
    classDef out fill:#FFF0DD,stroke:#C97822,color:#5A3212;
    class I input; class S,E1,E3 enc; class E2,NOTE attn; class D3,D2,D1,Q dec; class H,O out;
```

## 02_SA与Transformer展开.mmd

```mermaid
flowchart LR
    A["SA2 邻域输入<br/>每中心 K≤16 个 token×128"] --> B["LocalGeoPE<br/>[Δp/r, 距离/r, Δa]：7D<br/>MLP 7→128→128"]
    A --> C["特征边 MLP<br/>67→128→128"]
    B --> X(("相加")); C --> X
    X --> T["Pre-LN + 4头 MHA<br/>4×32D；邻域内注意力"] --> R1(("残差"))
    X -. "恒等支路" .-> R1
    R1 --> F["Pre-LN + FFN<br/>128→256→128；GELU"] --> R2(("残差"))
    R1 -. "恒等支路" .-> R2
    R2 --> P["Masked max pooling<br/>每中心输出128D"] --> V["SA2 输出<br/>125×128"]
    V --> N["PointNeXt-R InvRes<br/>局部聚合 + 128→256→128<br/>DropPath=0.10"]
    classDef a fill:#E8F1FB,stroke:#2878BD,color:#12304A;
    classDef g fill:#E1F3EF,stroke:#0B8F88,color:#123F3B;
    classDef t fill:#F0E5FA,stroke:#8650B5,color:#3C2055;
    class A,C,a; class B,g; class X,T,R1,F,R2,P,V,N t;
```

## 03_PointNeXt残差块展开.mmd

```mermaid
flowchart LR
    I["输入 F<br/>N×C"] --> L["局部 Ball 邻域<br/>[Δp,Fj]；max<br/>(C+3)→C→C"] --> A(("相加")) --> H["h<br/>N×C"]
    I -. "恒等支路" .-> A
    H --> P["倒置瓶颈 Pointwise MLP<br/>C→2C→C<br/>SA1：64→128→64<br/>SA2：128→256→128"] --> B(("相加")) --> O["输出 Fout<br/>N×C"]
    H -. "恒等支路" .-> B
    D["缩放因子 D<br/>SA1=0；SA2=0.10<br/>按病例共享；推理时D=1"] -.-> L
    D -.-> P
    classDef main fill:#E8F1FB,stroke:#2878BD,color:#12304A;
    classDef note fill:#FFF0DD,stroke:#C97822,color:#5A3212;
    class I,L,A,H,P,B,O main; class D note;
```

输入特征与逐层通道表见 [Excel](V5_R4_输入特征与通道表.xlsx)。

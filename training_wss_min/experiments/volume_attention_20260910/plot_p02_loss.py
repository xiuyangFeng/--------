"""Plot saved P02 training history; no evaluation or training is performed."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from training_wss_min.tools.volume_attention_common import EXP, RUNS, NAME, save_json, sha256, stamp

RUN = RUNS / NAME / "P02_s1234"
OUT = EXP / "P02_loss_diagnostic"


def mean_trailing(values, window=20):
    result = np.full(len(values), np.nan)
    result[window-1:] = np.convolve(values, np.ones(window)/window, mode="valid")
    return result


def main():
    OUT.mkdir(exist_ok=True)
    history_path = RUN / "history.jsonl"
    records = [json.loads(line) for line in history_path.read_text().splitlines()]
    assert [r["epoch"] for r in records] == list(range(400))
    cfg = json.loads((RUN / "config.json").read_text())
    epochs = np.arange(1, 401)
    loss = np.array([r["train_loss"] for r in records])
    mse = np.array([r["train_mse_norm"] for r in records])
    assert np.isfinite(loss).all() and (loss > 0).all()
    best = int(loss.argmin())
    # The logger records LR after the end-of-epoch scheduler step.
    first_lr = cfg["train"]["lr"] / max(1, cfg["train"]["warmup_epochs"])
    used_lr = np.array([first_lr] + [r["lr"] for r in records[:-1]])
    smooth = mean_trailing(loss)
    metrics = {ck: json.loads((RUN / "eval" / f"ckpt_{ck}" / "metrics.json").read_text())["test"]
               for ck in ("best", "last")}
    windows = []
    for a,b in ((1,50),(51,100),(101,150),(151,200),(201,250),(251,300),(301,350),(351,400),(351,375),(376,400)):
        x = loss[a-1:b]
        windows.append(dict(first_epoch=a, last_epoch=b, mean=float(x.mean()), median=float(np.median(x)),
                            std=float(x.std()), minimum=float(x.min()), sampled_mse_mean=float(mse[a-1:b].mean())))
    payload = dict(timestamp=stamp(),run=str(RUN),history_sha256=sha256(history_path),
                   epoch_display="one-based; history/ckpt use zero-based",
                   selection="minimum online epoch train_loss, averaged equally over batches; not test R2",
                   best_epoch=best+1,best_train_loss=float(loss[best]),last_epoch=400,last_train_loss=float(loss[-1]),
                   best_test_r2=metrics["best"]["field_casebalanced"]["r2"],last_test_r2=metrics["last"]["field_casebalanced"]["r2"],
                   smoothing="trailing arithmetic mean, 20 epochs; first19 values absent",windows=windows,
                   learning_rate="actual LR used: previous history.lr, first epoch initial warmup LR",
                   last_used_lr=float(used_lr[-1]),last_logged_next_lr=records[-1]["lr"],
                   limitation="No per-epoch test/validation curve; train loss is online with resampled points and stochastic training.")
    save_json(OUT / "loss_summary.json",payload)
    with (OUT / "loss_history.csv").open("w", newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["epoch_one_based","train_loss_batch_mean","train_mse_sampled_point_mean","trailing20_train_loss","lr_used","lr_logged_after_step","is_selected_best"])
        for i,r in enumerate(records):
            writer.writerow([i+1,float(loss[i]),float(mse[i]),float(smooth[i]) if np.isfinite(smooth[i]) else "",float(used_lr[i]),r["lr"],i==best])
    font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    font=FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"font.family":font,"font.size":11,"axes.unicode_minus":False,
                         "axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold",
                         "axes.labelcolor":"#334155","text.color":"#172B4D","pdf.fonttype":42})
    fig=plt.figure(figsize=(14.5,9.3),facecolor="white")
    gs=fig.add_gridspec(2,2,height_ratios=[1.25,1],left=.075,right=.97,bottom=.17,top=.865,wspace=.23,hspace=.43)
    full,tail,lr_ax,table_ax=[fig.add_subplot(gs[i,j]) for i,j in ((0,0),(0,1),(1,0),(1,1))]
    fig.suptitle("P02 压力实验：400轮训练损失与末段收敛",fontsize=21,fontweight="bold",x=.075,ha="left",y=.976)
    fig.text(.075,.919,"best按训练loss最低选择；last的测试R²更高，并不代表训练loss更低。",fontsize=13,color="#475569")
    blue,gray,orange,purple="#2563EB","#9AAABE","#C26719","#7C3AED"
    for ax in (full,tail):
        ax.plot(epochs,loss,color=gray,alpha=.72,lw=1,label="逐轮训练loss")
        ax.plot(epochs,smooth,color=blue,lw=2.3,label="20轮后向均值")
        ax.scatter(best+1,loss[best],marker="D",s=65,color=orange,zorder=5,label=f"best：第{best+1}轮")
        ax.scatter(400,loss[-1],s=60,color=purple,zorder=5,label="last：第400轮")
        ax.set_xlabel("训练轮次（从1计数）")
        ax.grid(axis="y",alpha=.2)
    full.set(title="A  全程：整体明显下降",ylabel="训练loss（标准化压力MSE，对数轴）",xlim=(1,406),ylim=(.026,1.08))
    full.set_yscale("log")
    full.set_yticks([.03,.05,.1,.2,.5,1.0],["0.03","0.05","0.10","0.20","0.50","1.00"])
    full.legend(loc="upper right",frameon=False,fontsize=10)
    tail.set(title="B  末80轮：平台附近仍有波动",ylabel="训练loss（线性轴）",xlim=(321,405),ylim=(.028,.070))
    assert loss[320:].max()<.070, "Tail plot must not hide recorded loss spikes"
    tail.axhline(loss[best],color=orange,ls=":",lw=1,alpha=.65)
    tail.annotate(f"best loss={loss[best]:.5f}",xy=(best+1,loss[best]),xytext=(328,.030),
                  arrowprops=dict(arrowstyle="-",color=orange),color=orange,fontsize=10)
    tail.annotate(f"last loss={loss[-1]:.5f}",xy=(400,loss[-1]),xytext=(359,.057),
                  arrowprops=dict(arrowstyle="-",color=purple),color=purple,fontsize=10)
    lr_ax.plot(epochs,used_lr,color="#0F766E",lw=2.2)
    lr_ax.axvspan(351,400,color="#E2E8F0",alpha=.55)
    lr_ax.set(title="C  学习率：末轮约1e-5",xlabel="训练轮次（从1计数）",ylabel="本轮更新使用的学习率",xlim=(1,406),yscale="log",ylim=(8e-6,1.3e-3))
    lr_ax.grid(axis="y",alpha=.2)
    lr_ax.text(.05,.13,"warmup 10轮 → 余弦下降\n灰色区域：最后50轮",transform=lr_ax.transAxes,fontsize=11,color="#475569")
    table_ax.axis("off")
    table_ax.set_title("D  两个已保存checkpoint的对照",loc="left",pad=15)
    data=[["轮次",str(best+1),"400"],
          ["在线训练loss",f"{loss[best]:.6f}",f"{loss[-1]:.6f}"],
          ["test34  R²_cb",f"{metrics['best']['field_casebalanced']['r2']:.6f}",f"{metrics['last']['field_casebalanced']['r2']:.6f}"],
          ["test34  RMSE / Pa",f"{metrics['best']['field']['rmse']:.3f}",f"{metrics['last']['field']['rmse']:.3f}"]]
    tab=table_ax.table(cellText=data,colLabels=["指标","best（按训练loss）","last"],colWidths=[.43,.34,.23],
                       loc="upper center",cellLoc="center",bbox=[0,.24,1,.76])
    tab.auto_set_font_size(False);tab.set_fontsize(10.6)
    for (rr,cc),cell in tab.get_celld().items():
        cell.set_edgecolor("#E2E8F0")
        cell.set_facecolor("#EAF0F8" if rr==0 else "#F8FAFC" if rr%2 else "white")
        if rr==0:cell.set_text_props(weight="bold")
    table_ax.text(0,.14,"只有best/last两次全量测试结果，无法画出每轮测试趋势。",transform=table_ax.transAxes,fontsize=10.5,color="#475569")
    table_ax.text(0,.025,"351–375轮均值 0.03487；376–400轮均值 0.03611。",transform=table_ax.transAxes,fontsize=10.5,color="#475569")
    fig.text(.075,.045,"口径：纯MSE，无PINN/尾部损失；训练点每轮重采样，loss为训练过程中各batch损失的均值。\n浅色逐轮值全部保留；平滑仅辅助读图。单seed1234，test34已暴露，仅作开发诊断。",fontsize=10.5,color="#64748B",va="bottom")
    fig.savefig(OUT / "P02_training_loss.png",dpi=190)
    fig.savefig(OUT / "P02_training_loss.pdf")
    plt.close(fig)
    analysis=("# P02损失曲线与延长训练的判断\n\n"
              f"best为第{best+1}轮：训练loss={loss[best]:.8f}；last为第400轮：loss={loss[-1]:.8f}。best按在线训练loss选，未按test R²选。\n\n"
              "301–350轮平均loss为0.039568，351–400轮为0.035491，说明最后100轮总体仍有下降；但351–375与376–400轮均值为0.034870/0.036113，最后50轮没有持续下降的证据。按全部采样点汇总的MSE同期仅由0.034231变为0.034137（下降0.27%），更接近平台。\n\n"
              "继续训练可能还有小幅收益，但现有曲线无法保证损失继续下降，更无法保证test R²提高。仅凭last优于best不足以证明400轮不够。训练loss含重采样、训练模式随机性和batch平均差异，且不是对最终checkpoint做固定训练点复评。\n\n"
              "如果后续验证延长训练，宜另登记同seed延长实验，保持400轮结果，比较固定评估口径的400/600轮，并明确学习率计划。按当前已降到约1e-5的学习率继续训练，与从零使用600轮余弦计划是不同实验；旧checkpoint仅保存模型等元数据、不含optimizer/scheduler/GradScaler完整状态，不能把加载旧权重称为无缝续训。本次仅画图和诊断，未执行额外训练。\n\n"
              "![P02损失曲线](P02_training_loss.png)\n\n[PDF](P02_training_loss.pdf) · [逐轮CSV](loss_history.csv) · [统计与口径](loss_summary.json)\n")
    (OUT / "README.md").write_text(analysis)
    print(json.dumps(payload,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()

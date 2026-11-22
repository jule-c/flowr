"""Generate a PQR file with OPLS4 partial charges for a target protein.

申请书正路：用 OPLS4 力场给靶蛋白口袋原子加偏电荷，供"原子在蛋白质电场
引导下的扩散"使用（公式(5)的电场由偏电荷叠加计算）。

产物 PQR 文件通过 generate_from_pdb.py 的 --pocket_charge_file 传入，
代码会自动按（链, 残基号, 残基名, 原子名）把电荷对齐到口袋原子。

流程（需要 Schrödinger 许可证，服务器上 $SCHRODINGER 环境变量已设置）：
  1) prepwizard: 补侧链/加氢/优化质子化状态            -> .mae
  2) mae_to_pqr.py（经 $SCHRODINGER/run 调用）：读取 .mae 中的 OPLS
     偏电荷并写成 PDB2PQR 风格 PQR（structconvert 不支持 PQR 输出）

用法（服务器 ~/flowr 下）：
  export PYTHONPATH=$PWD
  python scripts/add_opls4_charges.py \\
      --pdb_file gen_gpr119/GPR119_protein.pdb \\
      --out_pqr gen_gpr119/GPR119_opls4.pqr

常用 prepwizard 选项可通过 --prepwizard_args 透传，例如指定 pH 相关处理：
  --prepwizard_args "-fillsidechains -protassign -rehtreat -noprotassign"

备选方案（若 structconvert 导出的电荷全为 0，或没有 Schrödinger 许可证）：
  A) 在 Maestro 中对该结构应用 OPLS4（Protein Preparation Wizard 的最后
     一步 / Force Field 面板），然后 File > Export > PQR 导出；
  B) 用你们前期工作中已经建立的 OPLS4 加电荷管线（申请书数据部分），只要
     最终能导出每原子电荷的 PQR 即可；
  C) PDB2PQR（无许可证依赖；注意偏电荷来自 AMBER/PARSE 而非 OPLS4）：
     pdb2pqr30 --ff=AMBER --with-ph 7.4 --keep-chain --pdbinput in.pdb out.pqr

生成后请检查脚本打印的电荷统计：nonzero 比例和总电荷应明显非零，否则
电场引导不会起作用。
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flowr.util.partial_charges import pqr_charge_summary


def run(cmd, desc):
    print(f"[{desc}] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        [str(c) for c in cmd], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"{desc} failed with return code {result.returncode}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate an OPLS4-charged PQR for electrostatic field guidance."
    )
    parser.add_argument("--pdb_file", type=str, required=True, help="Input protein PDB")
    parser.add_argument(
        "--out_pqr", type=str, required=True, help="Output PQR file with partial charges"
    )
    parser.add_argument(
        "--prepwizard_args",
        type=str,
        default="-fillsidechains -WAIT",
        help="Extra flags for prepwizard (quoted string).",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Directory for the intermediate .mae file (default: alongside the output).",
    )
    args = parser.parse_args()

    schrodinger = os.environ.get("SCHRODINGER")
    if not schrodinger:
        raise RuntimeError(
            "The SCHRODINGER environment variable is not set. "
            "Point it to your Schrodinger installation, e.g. "
            "export SCHRODINGER=/opt/schrodinger2024-4"
        )

    pdb_file = Path(args.pdb_file).resolve()
    if not pdb_file.exists():
        raise FileNotFoundError(pdb_file)

    # Resolve to absolute paths: the Schrodinger job server runs the job in
    # its own working directory, where relative paths do not exist.
    out_pqr = Path(args.out_pqr).resolve()
    work_dir = (
        Path(args.work_dir).resolve() if args.work_dir else out_pqr.parent
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    mae_file = work_dir / (pdb_file.stem + "_prepped.mae")

    prepwizard = Path(schrodinger) / "utilities" / "prepwizard"

    # 1) Protein preparation
    run(
        [prepwizard] + args.prepwizard_args.split() + [str(pdb_file), str(mae_file)],
        "prepwizard",
    )
    if not mae_file.exists():
        raise RuntimeError(f"prepwizard did not produce {mae_file}")

    # 2) Convert to PQR. structconvert rejects the PQR extension (it has no
    # PQR writer), so call the bundled helper through the mmshare wrapper.
    helper = Path(__file__).resolve().parent / "mae_to_pqr.py"
    run(
        [Path(schrodinger) / "run", str(helper), str(mae_file), str(out_pqr)],
        "mae_to_pqr",
    )
    if not out_pqr.exists():
        raise RuntimeError(f"structconvert did not produce {out_pqr}")

    # 3) Sanity check the charges
    summary = pqr_charge_summary(out_pqr)
    print("\nPQR charge summary:")
    for key, val in summary.items():
        print(f"  {key}: {val}")

    n_atoms = summary.get("n_atoms", 0)
    n_nonzero = summary.get("n_nonzero", 0)
    if n_atoms == 0:
        raise RuntimeError("No atoms parsed from the PQR; check the file format.")
    if n_nonzero / n_atoms < 0.5:
        print(
            "\nWARNING: most charges are zero. The exported PQR probably does not "
            "contain force field partial charges. Apply OPLS4 in Maestro and "
            "export a PQR manually, or use your group's existing OPLS4 charging "
            "pipeline (see the alternatives in this script's docstring)."
        )
    else:
        print(
            f"\nDone. Use it with: --pocket_charge_file {out_pqr} "
            "(charges will be matched to the pocket atoms by name)."
        )


if __name__ == "__main__":
    main()

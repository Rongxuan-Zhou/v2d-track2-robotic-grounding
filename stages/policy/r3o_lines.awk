# r3o 队列行的核对（arm_var_chain.sh 用 awk -v oo=0|1 -f 本文件 读冒烟与正式队列的非注释行；契约 POLICY-I010）。
# oo=1（训练副本 v040_p185_nfg_oo）：每行要有 " V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=<σ> "、run 名带 -r3o-、两个变量各正好一次；
#   run 名带 -sgDD- 时 σ 必须等于 D.D（如 -sg04- ↔ 0.4、-sg20- ↔ 2.0），不带时 σ 必须是 0.4（t1o / t2o 段的取值）。
# oo=0（普通副本）：一行都不许带 V2D_OBJ_ORI_ 或 -r3o-。
# 输出：最后一行打坏行数（0 表示全过）；每条坏行先打 "BAD: <原因>: <行首 80 字符>"。
# 测试：stages/policy/test_r3o_lines.sh
{
  line = $0
  has = match(line, / V2D_OBJ_ORI_WEIGHT=1\.0 V2D_OBJ_ORI_STD=[0-9]+\.[0-9]+ /)
  std = ""
  if (has) { seg = substr(line, RSTART, RLENGTH); sub(/.*V2D_OBJ_ORI_STD=/, "", seg); sub(/ $/, "", seg); std = seg }
  any = index(line, "V2D_OBJ_ORI_") > 0
  named = (line ~ /-r3o-/)
  t = line; nw = gsub(/V2D_OBJ_ORI_WEIGHT=/, "", t); t = line; ns = gsub(/V2D_OBJ_ORI_STD=/, "", t)
  want = "0.4"
  if (match(line, /-sg[0-9][0-9]-/)) { tag = substr(line, RSTART + 3, 2); want = substr(tag, 1, 1) "." substr(tag, 2, 1) }
  reason = ""
  if (oo == 1) {
    if (!has) reason = "缺 V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=<σ>"
    else if (!named) reason = "run 名没有 -r3o-"
    else if (nw != 1 || ns != 1) reason = "两个变量要各正好一次"
    else if (std + 0 != want + 0 || std + 0 <= 0) reason = "σ " std " 与要求的 " want " 不符"
  } else if (any || named) reason = "普通副本的段不许带 V2D_OBJ_ORI_ 或 -r3o-"
  if (reason != "") { bad++; print "BAD: " reason ": " substr(line, 1, 80) }
}
END { print bad + 0 }

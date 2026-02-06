import json
import os
import matplotlib.pyplot as plt

def load_metrics(base_path, xs):
    result = {}
    for x in xs:
        json_path = os.path.join(base_path, str(x), "samples.json")
        if not os.path.exists(json_path):
            print(f"Warning: missing {json_path}")
            continue
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        result[x] = {
            "gen_ppl": data.get("generative_ppl"),
            "entropy": data.get("entropy")
        }
    return result

modelB_path = "./outputs/lm1b/duo_baseline_1024"
modelC_path = "./outputs/lm1b/duo_svg_1024-30000"
modelD_path = "./outputs/lm1b/temp_svg-50000"

steps = [8, 16, 32, 64, 128, 256]
x_pos = range(len(steps))

# modelA = load_metrics(modelA_path, steps)
modelB = load_metrics(modelB_path, steps)
modelC = load_metrics(modelC_path, steps)
modelD = load_metrics(modelD_path, steps)


# xA = list(modelA.keys())
# yA = [modelA[x]["gen_ppl"] for x in xA]
# eA = [modelA[x]["entropy"] for x in xA]

xB = list(modelB.keys())
yB = [modelB[x]["gen_ppl"] for x in xB]
eB = [modelB[x]["entropy"] for x in xB]

xC = list(modelC.keys())
yC = [modelC[x]["gen_ppl"] for x in xC]
eC = [modelC[x]["entropy"] for x in xC]

xD = list(modelD.keys())
yD = [modelD[x]["gen_ppl"] for x in xD]
eD = [modelD[x]["entropy"] for x in xD]

plt.figure(figsize=(8, 5))

# plt.plot(x_pos, yA, marker="o", label="MDLM", color='sandybrown', linestyle='--', alpha=0.7)
# for x, y, ent in zip(x_pos, yA, eA):
    # plt.text(x, y, f"({ent:.2f})", fontsize=9, ha='right', va='bottom')

plt.plot(x_pos, yB, marker="o", label="100000s baseline", color='cornflowerblue', alpha=0.9)
# for x, y, ent in zip(x_pos, yB, eB):
#     plt.text(x, y, f"({ent:.2f})", fontsize=9, ha='left', va='bottom')

plt.plot(x_pos, yC, marker="o", label="30000s svg", color='sandybrown', linestyle='--', alpha=0.7)
# for x, y, ent in zip(x_pos, yC, eC):
#     plt.text(x, y, f"({ent:.2f})", fontsize=9, ha='right', va='top')

plt.plot(x_pos, yD, marker="o", label="50000s svg", color='sandybrown', alpha=0.9)
# for x, y, ent in zip(x_pos, yD, eD):
#     plt.text(x, y, f"({ent:.2f})", fontsize=9, ha='left', va='top')

plt.xlabel("Sampling steps")
plt.ylabel("Generative PPL")
plt.title("Gen PPL (len=1024) vs Sampling Steps on LM1B Duo")
plt.xticks(x_pos, steps)

plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()


import os
if not os.path.exists("/home/jasonx62301/for_python/duo-svg/duo-svg/plot"):
    os.makedirs("/home/jasonx62301/for_python/duo-svg/duo-svg/plot")

output_path = "/home/jasonx62301/for_python/duo-svg/duo-svg/plot/faster_convergence_50000.png"
plt.savefig(output_path)

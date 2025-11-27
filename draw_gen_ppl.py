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



modelA_path = "./outputs/lm1b/baseline_gen_ppl_24000"
modelB_path = "./outputs/lm1b/svg_gen_ppl_24000"

steps = [8, 16, 32, 64, 128, 256]
x_pos = range(len(steps))

modelA = load_metrics(modelA_path, steps)
modelB = load_metrics(modelB_path, steps)



xA = list(modelA.keys())
yA = [modelA[x]["gen_ppl"] for x in xA]
eA = [modelA[x]["entropy"] for x in xA]

xB = list(modelB.keys())
yB = [modelB[x]["gen_ppl"] for x in xB]
eB = [modelB[x]["entropy"] for x in xB]



plt.figure(figsize=(8, 5))

plt.plot(x_pos, yA, marker="o", label="Baseline")
for x, y, ent in zip(x_pos, yA, eA):
    plt.text(x, y, f"({ent:.2f})", fontsize=9, ha='right', va='bottom')

plt.plot(x_pos, yB, marker="o", label="SVG")
for x, y, ent in zip(x_pos, yB, eB):
    plt.text(x, y, f"({ent:.2f})", fontsize=9, ha='left', va='bottom')


plt.xlabel("Sampling steps")
plt.ylabel("Generative PPL")
plt.title("Gen PPL and Entropy")
plt.xticks(x_pos, steps)

plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()


output_path = "/home/jasonx62301/for_python/duo/duo/plot/gen_ppl_24000.png"
plt.savefig(output_path)

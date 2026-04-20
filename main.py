#Alejandro Puerto Criado
#s2509433

from train import train
from evaluate import evaluate

print("1 Train")
print("2 Evaluate")

choice = input()

if choice == "1":
    train()
else:
    evaluate()


import random
import sys

N = int(sys.argv[1])
filename = sys.argv[2]

POINTS = set()

def distance(A, B):
  x1, y1 = A
  x2, y2 = B

  x = x1 - x2
  y = y1 - y2

  return str(x ** 2 + y ** 2)

if __name__ == "__main__":
  for _ in range(N):
    x = random.randint(1, 30)
    y = random.randint(1, 30)

    POINTS.add((x, y))

print(POINTS)

if len(list(POINTS)) != N:
  print("Less than", N, "cities generated. Exit 1")
  exit(1)

with open(filename, 'w+') as f:
  i = 0
  for p1 in list(POINTS):
    i += 1
    j = 0
    for p2 in list(POINTS):
      j += 1
      if i == j:
        f.write("-1")
      else:
        f.write(distance(p1, p2))
      
      if j != N:
        f.write(",")

    if i != N:
      f.write("\n")

    

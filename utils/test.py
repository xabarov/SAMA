import time

a = []
size = 10000000
start = time.process_time()
for i in range(size):
    a.append(i*i)

print(f"For time spent {time.process_time()-start}")

start = time.process_time()

b = [x*x for x in range(size)]

print(f"Gen time spent {time.process_time()-start}")


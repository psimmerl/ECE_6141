A  =  [0.3, 0.7]
B  =  [0.0, 0.0, 0.0]
C  =  [0.0, 0.0]
BA = [[0.3, 0.1], \
      [0.2, 0.4], \
      [0.5, 0.5]]
CB = [[0.2, 0.2, 0.4], \
      [0.8, 0.8, 0.6]]
AC = [[0.0, 0.0], \
      [0.0, 0.0]]

k = 0
for a in range(2):
    for b in range(3):
        for c in range(2):
            P = A[a]*BA[b][a]*CB[c][b]
            print("P(%s,%s,%s) = %0.3f" % (a, b, c, P))
            k += P

            B[b] += P
            C[c] += P
            AC[a][c] += P
        

print("Total = %.3f" % k)

for b in range(3):
    print("P(B=%s) = %.3f" % (b, B[b]))

for c in range(2):
    print("P(C=%s) = %.3f" % (c, C[c]))
     
for a in range(2):
    for c in range(2):
        print("P(A=%d, C=%d) = %0.3f" % (a, c, AC[a][c]))

for b in range(3):
    print("P(B=%d|C=%d) = %0.3f" % (b, 1, B[b]*CB[1][b]/C[1]))


for a in range(2):
    P = 0
    for b in range(3):
        P += A[a]*BA[b][a]*CB[c][b]/C[c]
    print("P(A=%d|C=%d) = %0.3f" % (a, 1, P))


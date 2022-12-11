using QPSReader

qps = readqps("..//ibell3a.mps")

li = qps.lvar
ui = qps.uvar
c0 = qps.c0

A = zeros(m,n)
b = zeros(m)

P_nnz = length(qps.qvals)
A_nnz = length(qps.avals)

for i = 1:P_nnz
    P[qps.qrows[i], qps.qcols[i]] = qps.qvals[i]
end

for i = 1:A_nnz
    A[qps.arows[i], qps.acols[i]] = qps.avals[i]
end

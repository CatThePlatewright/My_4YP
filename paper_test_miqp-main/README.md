# Readme
**Conic formulation:**

Conic programming (CP) formulation is:
$$
\min & \quad f(x,s,s_x) := \frac{1}{2}x^\top P x + q^\top x  \label{MICP-relaxation}\\
	\text{s.t} &\qquad \begin{bmatrix}
		A \\ -I
	\end{bmatrix} x + \begin{bmatrix}
		s \\ s_x
	\end{bmatrix} = \begin{bmatrix}
		b \\ 0
	\end{bmatrix}, \qquad \qquad (\text{CP}(l, u)) \notag\\
	&\qquad \qquad s \in \mathcal{C}, \notag\\
	& \qquad \qquad s_x \in \mathcal{B}, \notag
$$

After scale-ruiz operation, the scaled CP becomes:
$$
\min & \quad f(x,s,s_x) := \frac{1}{2}\hat{x}^\top \hat{P} \hat{x} + \hat{q}^\top \hat{x}  \label{scaled-MICP-relaxation}\\
	\text{s.t} &\qquad \begin{bmatrix}
		\hat{A} \\ -I
	\end{bmatrix} \hat{x} + \begin{bmatrix}
		\hat{s} \\ \hat{s}_x
	\end{bmatrix} = \begin{bmatrix}
		\hat{b} \\ 0
	\end{bmatrix}, \qquad \qquad (\text{CP}(l, u)) \notag\\
	&\qquad \qquad \hat{s} \in \hat{\mathcal{C}} :=E \cdot \mathcal{C}, \notag\\
	& \qquad \qquad D \hat{s}_x \in \mathcal{B}, \notag
$$
where $\hat{A} = EAD, \hat{b} = Eb$. The projection steps are implemented differently, i.e. projection onto $E \cdot \mathcal{C}$ and $D \hat{s}_x \in \mathcal{B}$.





## File explanation:

- **mpc_data/...:** generated mpc example from miosqp
- **data/...:** test data from "test_mpc.jl" and "test_randomQP.jl / final_test_randomQP.jl"
- **figure/...:** figures for the mpc test
- **src/...:**
  - **admm_operator.jl:**
  - **branch_and_bound.jl:** 



#### Additional files

- **cosmo_test.jl:** solve convex QP via COSMO

- **gurobi_test.jl:** solve MIQP
- **ToyExamples.jl:** QP-SDP



#### Experiments

with early termination vs. without early termination

- **test_mpc.jl:** test power electronics' case
- **test_randomQP.jl / final_test_randomQP.jl:** test random QP ()



#### Plots

- **plot_mpc.jl**


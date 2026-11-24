import HadamardTheorem5

/-!
# A kernel-checked witness for order 1948

The four rows are the published Goethals–Seidel sequences of Alpöge, Voinov,
and Reynolds-Haertle (https://hexagonmath.org/2611.00637v1, Table A1), also
recorded in `fixtures/published_1948.json`. Bit `i` in a row's mask indicates
that the sign at residue `i` is positive.
-/

namespace HadamardWitness1948

open Matrix BigOperators

private abbrev Residue := ZMod 487

private def signedRow (mask : Nat) (i : Residue) : ℤ :=
  if mask.testBit i.val then 1 else -1

private def a := signedRow
  0x1deb4aaa9b3ca7220de1d93872daabfb98ae37139e1e0529de35fef35528584cc7e4d267b2b6367ce297f5d6c2dafd618079506d07b22a408b764c3b68
private def b := signedRow
  0x768aa71177d524c192306915c7a9e2de76f05f46c2dd1b774225c67dcdee36f3385a31a2f1ffe8c063e00e7d3ce49254a5918d01390dd0aba7dca45227
private def c := signedRow
  0x1f88786a5634b8dbc4bf5c6c5f3a989b0197e8edfc266eec28ade690861bee54e78efdcbeab2c71fc9fd5a2d98160496ac79988067a5a378aa255765c0
private def d := signedRow
  0x5f9e78665272f16fa0bf4f6f7b3494be1913b8c73924ab7c26afebc8a8daa357f41dfbdfacba1734df9f0225913c011a3cd7389ea767827eb82544a59b

private theorem signedRow_sign (mask : Nat) (i : Residue) :
    signedRow mask i = 1 ∨ signedRow mask i = -1 := by
  unfold signedRow
  split <;> simp

private def term (i j : Residue) : ℤ :=
  a i * a j + b i * b j + c i * c j + d i * d j

private def paf (s : Residue) : ℤ :=
  ∑ i : Residue, term i (i + s)

private def multiplier : Residueˣ := ZMod.unitOfCoprime 362 (by decide)

private theorem term_swap (i j : Residue) : term i j = term j i := by
  unfold term
  rw [mul_comm (a i), mul_comm (b i), mul_comm (c i), mul_comm (d i)]

private theorem paf_neg (s : Residue) : paf (-s) = paf s := by
  unfold paf
  calc
    (∑ i : Residue, term i (i + -s)) =
        ∑ i : Residue, term (i + s) ((i + s) + -s) :=
      (Equiv.sum_comp (Equiv.addRight s) _).symm
    _ = ∑ i : Residue, term i (i + s) := by
      simp only [add_neg_cancel_right]
      exact Finset.sum_congr rfl fun i _ => term_swap (i + s) i

set_option maxRecDepth 8192 in
set_option maxHeartbeats 1000000 in
private theorem rows_fixed : ∀ i : Residue,
    a (multiplier * i) = a i ∧ b (multiplier * i) = b i ∧
      c (multiplier * i) = c i ∧ d (multiplier * i) = d i := by decide +kernel

private theorem term_mul (i j : Residue) :
    term (multiplier * i) (multiplier * j) = term i j := by
  obtain ⟨ahi, bhi, chi, dhi⟩ := rows_fixed i
  obtain ⟨ahj, bhj, chj, dhj⟩ := rows_fixed j
  simp only [term, ahi, bhi, chi, dhi, ahj, bhj, chj, dhj]

private theorem paf_mul (s : Residue) : paf (multiplier * s) = paf s := by
  unfold paf
  calc
    (∑ i : Residue, term i (i + multiplier * s)) =
        ∑ i : Residue, term (multiplier * i) (multiplier * i + multiplier * s) :=
      (Equiv.sum_comp multiplier.mulLeft _).symm
    _ = ∑ i : Residue, term i (i + s) := by
      simp only [← mul_add, term_mul]

private theorem paf_power (k : ℕ) (s : Residue) :
    paf ((-(multiplier : Residue)) ^ k * s) = paf s := by
  induction k with
  | zero => simp
  | succ k ih => rw [pow_succ', mul_assoc, neg_mul, paf_neg, paf_mul, ih]

private def representatives : List Residue := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 16, 17, 18, 19, 22, 27, 28, 29, 33, 37, 38, 54, 55, 76, 110]

-- Powers of -362 include both multiplication and shift reversal.
set_option maxRecDepth 8192 in
set_option maxHeartbeats 1000000 in
private theorem covers_shifts : ∀ s : Residue, s ≠ 0 →
    ∃ k : Fin 18, (-(multiplier : Residue)) ^ k.val * s ∈ representatives := by decide +kernel

set_option maxRecDepth 8192 in
set_option maxHeartbeats 4000000 in
private theorem paf_on_representatives :
    representatives.all (fun s => decide (paf s = 0)) = true := by decide +kernel

private theorem paf_zero (s : Residue) (hs : s ≠ 0) : paf s = 0 := by
  obtain ⟨k, hk⟩ := covers_shifts s hs
  rw [← paf_power k.val s]
  exact of_decide_eq_true (List.all_eq_true.mp paf_on_representatives _ hk)

/-- The published order-1948 witness satisfies the full Hadamard identity. -/
theorem hadamard_order_1948 :
    ∃ M : Matrix (Fin 1948) (Fin 1948) ℤ,
      (∀ i j, M i j = 1 ∨ M i j = -1) ∧
      M * Mᵀ = (1948 : ℤ) • (1 : Matrix (Fin 1948) (Fin 1948) ℤ) := by
  let e := ZMod.finEquiv 487
  refine ⟨HadamardTheorem5.H_unbordered (a ∘ e) (b ∘ e) (c ∘ e) (d ∘ e), ?_⟩
  refine HadamardTheorem5.goethals_seidel_hadamard (a ∘ e) (b ∘ e) (c ∘ e) (d ∘ e)
    (fun i => signedRow_sign _ (e i)) (fun i => signedRow_sign _ (e i))
    (fun i => signedRow_sign _ (e i)) (fun i => signedRow_sign _ (e i)) ?_
  intro s hs
  change (∑ i : Fin 487, term (e i) (e (i + s))) = 0
  simp only [map_add]
  exact (Equiv.sum_comp e.toEquiv (fun i => term i (i + e s))).trans
    (paf_zero (e s) (fun he => hs (e.injective (he.trans e.map_zero.symm))))

end HadamardWitness1948

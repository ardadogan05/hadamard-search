import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Group.Equiv.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

/-!
# The split order-three obstruction for the width-12 construction

For the split action on `(ℤ/2ℤ)² × S`, each invariant sequence has four
cosets of the form `(x, y, y, y)`. This file certifies the obstruction
from the single-shift autocorrelation, quotient norm, and total row norm.
-/

namespace HadamardSigma3

open BigOperators

private lemma square_le_three {d : ℤ} (h : d ^ 2 ≤ 3) : -1 ≤ d ∧ d ≤ 1 := by
  constructor
  · by_contra! hd
    have : d ≤ -2 := by omega
    nlinarith [mul_nonneg (show 0 ≤ -d - 2 by omega) (show 0 ≤ -d + 2 by omega)]
  · by_contra! hd
    have : d ≥ 2 := by omega
    nlinarith [mul_nonneg (show 0 ≤ d - 2 by omega) (show 0 ≤ d + 2 by omega)]

private lemma reduced_order (q d : Fin 4 → ℤ) (s n : ℤ) (j : Fin 4)
    (hqj : q j ^ 2 = 1) (hq : ∀ i, i ≠ j → q i = 0)
    (hmod : ∀ i, (q i - d i) % 2 = s % 2)
    (hd : ∑ i, d i ^ 2 = 3)
    (hn : ∑ i, (4 * q i - 2 * d i) ^ 2 = n) : n = 28 := by
  classical
  have hsplit : (∑ i ∈ Finset.univ.erase j, d i ^ 2) + d j ^ 2 = 3 := by
    rw [Finset.sum_erase_add _ _ (Finset.mem_univ j), hd]
  have hcard : (Finset.univ.erase j).card = 3 := by simp
  have hbound (i : Fin 4) : -1 ≤ d i ∧ d i ≤ 1 := by
    apply square_le_three
    rw [← hd]
    exact Finset.single_le_sum (fun k _ => sq_nonneg (d k)) (Finset.mem_univ i)
  have hs : s % 2 = 1 := by
    by_contra h
    have hs0 : s % 2 = 0 := by omega
    have hzero : (∑ i ∈ Finset.univ.erase j, d i ^ 2) = 0 := by
      apply Finset.sum_eq_zero
      intro i hi
      have he := hmod i
      rw [hq i (Finset.mem_erase.mp hi).1, hs0] at he
      have hb := hbound i
      have : d i = 0 := by omega
      simp [this]
    rw [hzero] at hsplit
    have hb := hbound j
    have : d j = -1 ∨ d j = 0 ∨ d j = 1 := by omega
    rcases this with h | h | h <;> norm_num [h] at hsplit
  have hd_other (i : Fin 4) (hi : i ≠ j) : d i ^ 2 = 1 := by
    have he := hmod i
    rw [hq i hi, hs] at he
    have hb := hbound i
    have : d i = -1 ∨ d i = 1 := by omega
    rcases this with h | h <;> norm_num [h]
  have hsum : (∑ i ∈ Finset.univ.erase j, d i ^ 2) = 3 := by
    calc
      _ = ∑ _i ∈ Finset.univ.erase j, (1 : ℤ) := by
        apply Finset.sum_congr rfl
        intro i hi
        exact hd_other i (Finset.mem_erase.mp hi).1
      _ = 3 := by simp [hcard]
  rw [hsum] at hsplit
  have hdj : d j = 0 := by
    have hb := hbound j
    have : d j = -1 ∨ d j = 0 ∨ d j = 1 := by omega
    rcases this with h | h | h
    · norm_num [h] at hsplit
    · exact h
    · norm_num [h] at hsplit
  have hrow : (∑ i ∈ Finset.univ.erase j, (4 * q i - 2 * d i) ^ 2) = 12 := by
    calc
      _ = ∑ _i ∈ Finset.univ.erase j, (4 : ℤ) := by
        apply Finset.sum_congr rfl
        intro i hi
        have hi' := (Finset.mem_erase.mp hi).1
        rw [hq i hi']
        nlinarith only [hd_other i hi']
      _ = 12 := by simp [hcard]
  rw [← Finset.sum_erase_add _ _ (Finset.mem_univ j), hrow, hdj] at hn
  nlinarith only [hn, hqj]

/-- Four split sequences, with `x` on the zero coset and `y` on each of
the three nonzero cosets. The first identity is the autocorrelation at any
nonzero shift in the factor `(ℤ/2ℤ)²` with zero shift in `S`. The other two
identities are the squared norms of the nontrivial quotient component and the
total row sums, respectively. -/
theorem order_eq_28 {S : Type*} [Fintype S] (x y : Fin 4 → S → ℤ) (n : ℤ)
    (hx : ∀ i z, x i z = 1 ∨ x i z = -1)
    (hy : ∀ i z, y i z = 1 ∨ y i z = -1)
    (hshift : ∑ i, (∑ z, (2 + 2 * x i z * y i z)) = 4)
    (hquot : ∑ i, ((∑ z, x i z) - (∑ z, y i z)) ^ 2 = 12)
    (hrow : ∑ i, ((∑ z, x i z) + 3 * (∑ z, y i z)) ^ 2 = n) :
    n = 28 := by
  classical
  let e (i : Fin 4) : ℤ := ∑ z, if x i z = y i z then 1 else 0
  let q (i : Fin 4) : ℤ := ∑ z, if x i z = y i z then x i z else 0
  let d (i : Fin 4) : ℤ := ∑ z, if x i z = y i z then 0 else x i z
  have he_nonneg (i : Fin 4) : 0 ≤ e i := by
    apply Finset.sum_nonneg
    intro z _
    split <;> norm_num
  have hxy (i : Fin 4) :
      (∑ z, x i z) = q i + d i ∧ (∑ z, y i z) = q i - d i := by
    constructor
    · dsimp only [q, d]
      rw [← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro z _
      split <;> simp
    · dsimp only [q, d]
      rw [← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl
      intro z _
      rcases hx i z with h | h <;> rcases hy i z with h' | h' <;> simp [h, h']
  have hshift_eq (i : Fin 4) : (∑ z, (2 + 2 * x i z * y i z)) = 4 * e i := by
    dsimp only [e]
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro z _
    rcases hx i z with h | h <;> rcases hy i z with h' | h' <;> norm_num [h, h']
  simp_rw [hshift_eq, ← Finset.mul_sum] at hshift
  have he_sum : ∑ i, e i = 1 := by omega
  have hex : ∃ j, e j ≠ 0 := by
    by_contra! h
    simp [h] at he_sum
  obtain ⟨j, hj⟩ := hex
  have hej : e j = 1 := by
    have hle := Finset.single_le_sum (fun i _ => he_nonneg i) (Finset.mem_univ j)
    have hn := he_nonneg j
    omega
  have he_other (i : Fin 4) (hi : i ≠ j) : e i = 0 := by
    have hsplit := Finset.sum_erase_add Finset.univ e (Finset.mem_univ j)
    have hle := Finset.single_le_sum (fun k _ => he_nonneg k)
      (Finset.mem_erase.mpr ⟨hi, Finset.mem_univ i⟩)
    have hn := he_nonneg i
    omega
  have hq_decomp (i : Fin 4) :
      q i = e i - 2 * ∑ z, (if x i z = y i z ∧ x i z = -1 then 1 else 0 : ℤ) := by
    dsimp only [q, e]
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro z _
    rcases hx i z with h | h <;> rcases hy i z with h' | h' <;> norm_num [h, h']
  have hq_bound (i : Fin 4) : -e i ≤ q i ∧ q i ≤ e i := by
    constructor
    · dsimp only [q, e]
      rw [← Finset.sum_neg_distrib]
      apply Finset.sum_le_sum
      intro z _
      rcases hx i z with h | h <;> rcases hy i z with h' | h' <;> norm_num [h, h']
    · apply Finset.sum_le_sum
      intro z _
      rcases hx i z with h | h <;> rcases hy i z with h' | h' <;> norm_num [h, h']
  have hqj : q j ^ 2 = 1 := by
    have hb := hq_bound j
    have hp := hq_decomp j
    rw [hej] at hb hp
    have : q j = -1 ∨ q j = 1 := by omega
    rcases this with h | h <;> norm_num [h]
  have hq_other (i : Fin 4) (hi : i ≠ j) : q i = 0 := by
    have hb := hq_bound i
    rw [he_other i hi] at hb
    omega
  have hmod (i : Fin 4) : (q i - d i) % 2 = (Fintype.card S : ℤ) % 2 := by
    have hsum : (∑ z, y i z) = (Fintype.card S : ℤ) -
        2 * ∑ z, (if y i z = -1 then 1 else 0 : ℤ) := by
      have hcard : (∑ _z : S, (1 : ℤ)) = (Fintype.card S : ℤ) := by simp
      rw [← hcard, Finset.mul_sum, ← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl
      intro z _
      rcases hy i z with h | h <;> norm_num [h]
    rw [(hxy i).2] at hsum
    omega
  have hquot' : ∑ i, d i ^ 2 = 3 := by
    have hscale (i : Fin 4) : ((∑ z, x i z) - (∑ z, y i z)) ^ 2 =
        4 * (d i ^ 2) := by rw [(hxy i).1, (hxy i).2]; ring
    simp_rw [hscale, ← Finset.mul_sum] at hquot
    omega
  have hrow' : ∑ i, (4 * q i - 2 * d i) ^ 2 = n := by
    have hr (i : Fin 4) : (∑ z, x i z) + 3 * (∑ z, y i z) =
        4 * q i - 2 * d i := by rw [(hxy i).1, (hxy i).2]; ring
    simpa only [hr] using hrow
  exact reduced_order q d (Fintype.card S) n j hqj hq_other hmod hquot' hrow'

/-- With the order parameter of the width-12 construction, the split
identities force the subgroup to have exactly one element. -/
theorem card_eq_one {S : Type*} [Fintype S] (x y : Fin 4 → S → ℤ)
    (hx : ∀ i z, x i z = 1 ∨ x i z = -1)
    (hy : ∀ i z, y i z = 1 ∨ y i z = -1)
    (hshift : ∑ i, (∑ z, (2 + 2 * x i z * y i z)) = 4)
    (hquot : ∑ i, ((∑ z, x i z) - (∑ z, y i z)) ^ 2 = 12)
    (hrow : ∑ i, ((∑ z, x i z) + 3 * (∑ z, y i z)) ^ 2 =
      16 * (Fintype.card S : ℤ) + 12) :
    Fintype.card S = 1 := by
  have h := order_eq_28 x y _ hx hy hshift hquot hrow
  omega


section FullAutocorrelation

variable {S : Type*} [Fintype S] [AddGroup S]

/-- Periodic cross-correlation on the second factor of the split group. -/
private def corr (x y : S → ℤ) (u : S) : ℤ := ∑ z, x z * y (z + u)

private lemma sum_corr (x y : S → ℤ) :
    (∑ u, corr x y u) = (∑ z, x z) * (∑ z, y z) := by
  unfold corr
  rw [Finset.sum_comm, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro z _
  rw [← Finset.mul_sum]
  congr 1
  simpa using Equiv.sum_comp (Equiv.addLeft z) y

/-- Combined autocorrelation at shift `(0, u)` for the split rows. -/
def zeroCosetPAF (x y : Fin 4 → S → ℤ) (u : S) : ℤ :=
  ∑ i, (corr (x i) (x i) u + 3 * corr (y i) (y i) u)

/-- Combined autocorrelation at shift `(c, u)`, for a nonzero `c` in
`(ℤ/2ℤ)²`. Addition by `c` swaps the zero and `c` cosets, and swaps the
two remaining cosets. -/
def nonzeroCosetPAF (x y : Fin 4 → S → ℤ) (u : S) : ℤ :=
  ∑ i, (corr (x i) (y i) u + corr (y i) (x i) u + 2 * corr (y i) (y i) u)

/-- The required autocorrelations of the split width-12 family force
`|S| = 1`, and hence the construction has order 28. -/
theorem card_eq_one_of_paf (x y : Fin 4 → S → ℤ)
    (hx : ∀ i z, x i z = 1 ∨ x i z = -1)
    (hy : ∀ i z, y i z = 1 ∨ y i z = -1)
    (hzero : ∀ u, u ≠ 0 → zeroCosetPAF x y u = -12)
    (hnonzero : ∀ u, nonzeroCosetPAF x y u = 4) :
    Fintype.card S = 1 := by
  classical
  let s : ℤ := Fintype.card S
  have hP0 : zeroCosetPAF x y 0 = 16 * s := by
    have hself (i : Fin 4) (z : S) : x i z * x i z = 1 ∧ y i z * y i z = 1 := by
      constructor
      · rcases hx i z with h | h <;> norm_num [h]
      · rcases hy i z with h | h <;> norm_num [h]
    have hxself (i : Fin 4) : corr (x i) (x i) 0 = s := by
      simp [corr, fun z => (hself i z).1, s]
    have hyself (i : Fin 4) : corr (y i) (y i) 0 = s := by
      simp [corr, fun z => (hself i z).2, s]
    simp only [zeroCosetPAF, hxself, hyself, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul]
    ring
  have hshift : ∑ i, (∑ z, (2 + 2 * x i z * y i z)) = 4 := by
    rw [← hnonzero 0, nonzeroCosetPAF]
    apply Finset.sum_congr rfl
    intro i _
    simp only [corr, add_zero, Finset.mul_sum, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro z _
    rcases hx i z with h | h <;> rcases hy i z with h' | h' <;> norm_num [h, h']
  have hsum0 : ∑ u, zeroCosetPAF x y u = 4 * s + 12 := by
    rw [← Finset.sum_erase_add _ _ (Finset.mem_univ (0 : S)), hP0]
    have hc : ((Finset.univ.erase (0 : S)).card : ℤ) = s - 1 := by
      rw [Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ]
      have hnonempty : 0 < Fintype.card S := Fintype.card_pos_iff.mpr ⟨(0 : S)⟩
      omega
    rw [show (∑ u ∈ Finset.univ.erase (0 : S), zeroCosetPAF x y u) =
        ∑ _u ∈ Finset.univ.erase (0 : S), (-12 : ℤ) from
      Finset.sum_congr rfl (fun u hu => hzero u (Finset.mem_erase.mp hu).1)]
    simp only [Finset.sum_const, nsmul_eq_mul, hc]
    ring
  have hsum1 : ∑ u, nonzeroCosetPAF x y u = 4 * s := by
    simp only [hnonzero, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, s]
    ring
  let a (i : Fin 4) : ℤ := ∑ z, x i z
  let b (i : Fin 4) : ℤ := ∑ z, y i z
  have ha : ∑ i, (a i * a i + 3 * (b i * b i)) = 4 * s + 12 := by
    rw [← hsum0]
    unfold zeroCosetPAF
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro i _
    simp only [Finset.sum_add_distrib, ← Finset.mul_sum, sum_corr, a, b]
  have hb : ∑ i, (a i * b i + b i * a i + 2 * (b i * b i)) = 4 * s := by
    rw [← hsum1]
    unfold nonzeroCosetPAF
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro i _
    simp only [Finset.sum_add_distrib, ← Finset.mul_sum, sum_corr, a, b]
  have hquot : ∑ i, (a i - b i) ^ 2 = 12 := by
    calc
      _ = (∑ i, (a i * a i + 3 * (b i * b i))) -
          ∑ i, (a i * b i + b i * a i + 2 * (b i * b i)) := by
        rw [← Finset.sum_sub_distrib]
        apply Finset.sum_congr rfl
        intro i _
        ring
      _ = 12 := by rw [ha, hb]; ring
  have hrow : ∑ i, (a i + 3 * b i) ^ 2 = 16 * s + 12 := by
    calc
      _ = (∑ i, (a i * a i + 3 * (b i * b i))) +
          3 * ∑ i, (a i * b i + b i * a i + 2 * (b i * b i)) := by
        rw [Finset.mul_sum, ← Finset.sum_add_distrib]
        apply Finset.sum_congr rfl
        intro i _
        ring
      _ = _ := by rw [ha, hb]; ring
  exact card_eq_one x y hx hy hshift hquot hrow

end FullAutocorrelation

end HadamardSigma3

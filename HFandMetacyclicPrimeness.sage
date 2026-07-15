# HFandMetacyclicPrimeness.sage
# Runs the full primeness-testing procedure of Section 10.2 ("The general
# approach") on a single, given knot, using the various functions found below in the "computation" section. This test: enumerate positive-symmetric
# factorizations, prunes with Jones/HOMFLY, prunes with homology-order
# consistency on p-fold branched covers, then prunes with metacyclic
# Betti-number obstructions (both the d1 != d2 and d1 = d2 cases).
#
# Usage:
#   load('test_generic_knot.sage')
#   result = HFandMetaPrimeTest([[4,2,5,1],[8,6,1,5],[6,3,7,4],[2,7,3,8]])   # PD code
#   result = HFandMetaPrimeTest('DT: [(4,6,2)]')                            # DT code
#   result = HFandMetaPrimeTest(pd, primes=[2,3,5,7,11,13], verbose=True)   # go further

# ============= Imports =============

import snappy
from sympy import symbols #for HF polynomials, polynomial rings
import numpy as np #for polynomial division

# ============= Testing =============

def _pd_from_code(code):
    # Normalize a PD code (list of 4-lists) or DT code (string) to a
    # canonical PD code via SnapPy, so the rest of the pipeline always
    # sees a well-formed, consistently-indexed PD code.
    if isinstance(code, str):
        s = code.strip()
        if not s.upper().startswith("DT"):
            s = "DT: " + s
        K = snappy.Link(s)
    else:
        K = snappy.Link(code)
    return [list(x) for x in K.PD_code()]


def _step1_factorizations(pd):
    # Step 1: enumerate positive-symmetric factorizations Omega_K = Omega_1*Omega_2,
    # drop the trivial one, require non-negative coefficients,
    # then prune with the Jones and HOMFLY polynomial tests.
    hf_poly = hf_polynomial(pd)[0]
    facs = all_sym_factor_pairs(hf_poly)
    facs = [x for x in facs if 1 not in x]
    if len(facs) == 0:
        return []
    facs = [x for x in facs if check_pos_coeff(x[0]) and check_pos_coeff(x[1])]
    if len(facs) == 0:
        return []
    K = snappy.Link(pd)
    facs = [fac for fac in facs if not Jones_test(K, fac)]
    if len(facs) == 0:
        return []
    facs = [fac for fac in facs if HOMFLY_TEST(pd, fac)]
    return facs


def _step2_homology_order(pd, facs, primes):
    # Step 2: for each cover order p, require that the orders of H_1(Xp(K1))
    # and H_1(Xp(K2)) implied by each factor are consistent with the actual H_1(Xp(K)).
    for p in primes:
        hom_group = cyclic_branched_cover_homology(pd, p)
        allowed = orders_of_subgroups(hom_group)
        facs = [
            fac for fac in facs
            if poly_to_order_hom_sqrt(fac[0], p) in allowed
            and poly_to_order_hom_sqrt(fac[1], p) in allowed
        ]
        if len(facs) == 0:
            return facs, p
    return facs, None


def _cover_orders(fac, p, Rs, Rt):
    # |H_1(Xp(K1))| and |H_1(Xp(K2))| computed directly from the two factor
    # polynomials: for p=2, evaluate at (s,t)=(-1,-1); for odd p, take the
    # resultant of the Alexander polynomial with t^p - 1.
    s_, t_ = Rs.gens()
    if p == 2:
        return abs(int(fac[0].subs({s_: -1, t_: -1}))), abs(int(fac[1].subs({s_: -1, t_: -1})))
    t = Rt.gen()
    alex1 = Rt(fac[0].subs({s_: -1}))
    alex2 = Rt(fac[1].subs({s_: -1}))
    return abs(int(alex1.resultant(t**p - 1, t))), abs(int(alex2.resultant(t**p - 1, t)))


def _steps3to5_metacyclic(pd, facs, primes):
    # Steps 3-5: for each cover order p and each remaining factorization,
    # look for prime pairs (d1,d2) dividing |H_1(Xp(K1))|, |H_1(Xp(K2))|
    # and run the matching metacyclic obstruction test. A factorization is
    # dropped as soon as one obstruction rules it out.
    Rs = PolynomialRing(QQ, "s,t")
    Rt = PolynomialRing(ZZ, "t")
    remaining = []
    for fac in facs:
        obstructed = False
        for p in primes:
            order1, order2 = _cover_orders(fac, p, Rs, Rt)
            if order1 == 0 or order2 == 0:
                continue
            primes1 = [q for q, _ in factor(order1)]
            primes2 = [q for q, _ in factor(order2)]

            # steps 3-4: d1 != d2
            for d1 in primes1:
                if obstructed:
                    break
                for d2 in primes2:
                    if d1 == d2 or order1 % d2 == 0 or order2 % d1 == 0:
                        continue
                    if len(find_p_roots_mod_q(d1, p)) == 0 or len(find_p_roots_mod_q(d2, p)) == 0:
                        continue
                    if not metacyclic_test_diff_primes(pd, p, d1, d2):
                        obstructed = True
                        break
            if obstructed:
                break

            # step 5: d1 = d2 = d, with d^2 (d=2) or d^4 (d odd) dividing neither order
            for d in set(primes1) & set(primes2):
                bound = d**2 if d == 2 else d**4
                if order1 % bound == 0 or order2 % bound == 0:
                    continue
                if len(find_p_roots_mod_q(d, p)) == 0:
                    continue
                if not metacyclic_test_same_prime(pd, p, d):
                    obstructed = True
                    break
            if obstructed:
                break

        if not obstructed:
            remaining.append(fac)
    return remaining


def HFandMetaPrimeTest(code, primes=[2, 3, 5, 7], verbose=False):
    # Run the full Section 10.2 procedure on a single knot. `code` is a PD
    # code (list of 4-lists) or a DT code (string). `primes` is the list of
    # cover orders p to use in steps 2-5; default is with [2,3,5,7].
    #
    # Returns a dict:
    #   prime:    True if K was confirmed prime, None if undetermined
    #             (these tests never prove compositeness, only primeness),
    #   stage:    which test confirmed primeness, or "inconclusive",
    #   pd:       the (normalized) PD code that was tested,
    #   remaining_factorizations: the positive-symmetric factorizations
    #             Omega_1*Omega_2 that survived every test, if any.
    def done(is_prime, stage, remaining):
        if verbose:
            status = "prime" if is_prime else "undetermined"
            tail = "" if is_prime else f" -- {len(remaining)} factorization(s) survive"
            print(f"[{stage}] {status}{tail}")
        return {"prime": is_prime or None, "stage": stage, "pd": pd, "remaining_factorizations": remaining}

    pd = _pd_from_code(code)

    facs = _step1_factorizations(pd)
    if len(facs) == 0:
        return done(True, "step1_factorization_jones_homfly", [])

    facs, failed_at = _step2_homology_order(pd, facs, primes)
    if len(facs) == 0:
        return done(True, f"step2_homology_order_p{failed_at}", [])

    facs = _steps3to5_metacyclic(pd, facs, primes)
    if len(facs) == 0:
        return done(True, "steps3-5_metacyclic", [])

    return done(False, "inconclusive", facs)


def is_prime_knot(code, primes=[2, 3, 5, 7]):
    # Convenience wrapper: True if confirmed prime, else None (undetermined).
    return HFandMetaPrimeTest(code, primes=primes)["prime"]


# ============= Computation =============


# ---------------------------------------------------------------------
# 1. PD code -> Wirtinger presentation -> Fox matrix
# ---------------------------------------------------------------------

def pd_to_arc(pd):
    # Group pd's edges into arcs (maximal runs uninterrupted by an undercrossing).
    cr= len(pd)
    mod = 2*cr
    pd_in = [[(y % mod) for y in x] for x in pd]
    over_edges = [ [x[1],x[3]] for x in pd_in ]
    over_edges_sort = []
    for i in range(cr):
        if (over_edges[i][1] - over_edges[i][0] == 1):
            over_edges_sort.append(over_edges[i])
        elif (over_edges[i][1] - over_edges[i][0] == -1):
            over_edges_sort.append([over_edges[i][1],over_edges[i][0]])
        else:
            over_edges_sort.append([max(over_edges[i]), min(over_edges[i])])
    overs = sorted(over_edges_sort)
    for i in range(cr - 1):
        ind = cr - i - 1
        if overs[ind][0] == overs[ind - 1][1]:
            overs[ind - 1] = overs[ind - 1] + overs[ind]
            overs = overs[:ind] + overs[ind+1:]
    ovcount = len(overs)
    if (overs[ovcount - 1][1] == overs[0][0]):
        overs[0] = overs[ovcount -1] + overs[0]
        overs = overs[:ovcount - 1]
    overs = [ list(set(x)) for x in overs]
    if overs[len(overs) - 1][1] == overs[0][0]:
        overs[0] = list(set(overs[0] + overs[len(overs) - 1]))
        overs = overs[:len(overs) - 1]
    indices = flatten(overs)
    for i in range(mod):
        if i not in indices:
            overs.append([i])
    overs = [tuple(sorted(x)) for x in overs]
    overs = sorted(overs)
    leng = len(overs)
    for i in range(leng - 1):
        over2 = overs[leng - i - 1]
        over1 = overs[leng - i - 2]
        if bool(  set(over2) & set(over1)):
            overs[leng - i - 2] =   tuple(sorted(list(set(list(over1) + list(over2)))))
            overs =  overs[:leng - i - 1] + overs[leng - i:]
    return sorted(overs)

def dict_arc_edge(pd):
    # Build edge<->arc index dictionaries from pd_to_arc.
    overs = pd_to_arc(pd)
    dict_arc_to_ed = {}
    for i in range(len(pd)):
        dict_arc_to_ed[i] = overs[i]
    dict_in = dict_arc_to_ed
    reversed_dict = {num: key for key, value in dict_in.items() for num in value}
    reversed_dict1 = dict(sorted(reversed_dict.items()))
    return [dict_arc_to_ed, reversed_dict1]

def wirtinger_presentation(pd_in):
    # Wirtinger presentation of the knot group: relators [a,b,c] mean x_a x_b x_a^-1 x_c^-1.
    cr = len(pd_in)
    mod = 2*cr
    pd = [[(y % mod) for y in x] for x in pd_in]
    dict_e_to_a = dict_arc_edge(pd)[1]
    wirt_e = []
    for i in range(cr):
        pdi = pd[i]
        x0 = pdi[0]
        x1 = pdi[1]
        x2 = pdi[2]
        x3 = pdi[3]
        if (x3 - x1 == 1 or x3 - x1 < -1):
            wirt_e.append([x1, x0 ,x2])
        elif (x3 - x1 == -1 or x3 - x1 > 1):
            wirt_e.append([x1, x2, x0])
    wirt_a = []
    for i in range(cr):
        a = wirt_e[i][0]
        b = wirt_e[i][1]
        c = wirt_e[i][2] 
        wirt_a.append( [dict_e_to_a[a], dict_e_to_a[b], dict_e_to_a[c] ] )
    return wirt_a

def fox_matrix(pd):
    # Fox derivative matrix for d2: C_2 -> C_1, entries in Z[free group on t_i].
    cr = len(pd)
    wirt = wirtinger_presentation(pd)
    variable_names = [f"t{i}" for i in range(cr)]
    Rtemp = PolynomialRing(QQ, names=variable_names)
    t = Rtemp.gens()  
    FoxM  = zero_matrix(Rtemp, cr - 1, cr)
    for i in range(cr - 1):
        a = wirt[i][0]
        b = wirt[i][1]
        c = wirt[i][2]
        FoxM[i , a] =  1 - t[c]
        FoxM[i,b] =   t[a]
        FoxM[ i, c] =  -1
    return FoxM

def fox_matrix0(pd):
    # Fox derivative matrix for d1: C_1 -> C_0.
    cr = len(pd)
    variable_names = [f"t{i}" for i in range(cr)]
    Rtemp = PolynomialRing(QQ, names=variable_names)
    t = Rtemp.gens()  
    FoxM0  = zero_matrix(Rtemp, cr , 1)
    for i in range(cr ):
        FoxM0[i , 0] =  1 - t[i]
    return FoxM0


# ---------------------------------------------------------------------
# 2. Representations and twisted homology via Fox calculus
# ---------------------------------------------------------------------

def substitute_polynomial_with_matrices2(F, matrices):
    # Substitute a list of matrices for polynomial F's variables, returning the resulting matrix.
    k = matrices[0].nrows()
    ring = matrices[0][0,0].parent()
    result = zero_matrix(ring, k, k)
    for monomial, coefficient in F.dict().items():
        term_matrix = identity_matrix(ring, k)
        for var_index, power in enumerate(monomial):
            if power > 0:
                term_matrix *= matrices[var_index]**power
    
        result += ring(coefficient) * term_matrix
    return result

def fox_matrix_rep(fox, matrices):
    # Apply a representation (list of matrices) entrywise to a Fox matrix, returning the resulting block matrix.
    dim_rep = matrices[0].nrows()
    dim_fox =(fox.nrows(), fox.ncols())
    ring = matrices[0][0,0].parent()
    bigmat = matrix(ring, dim_rep*dim_fox[0], dim_rep*(dim_fox[1] ))
    for rf in range(dim_fox[0]):
        for cf in range(dim_fox[1]):
            fox_ent = fox[rf, cf]
            mat_to_enter = substitute_polynomial_with_matrices2(fox_ent, matrices)
            for rrep in range(dim_rep):
                for crep in range(dim_rep):
                    bigmat[rf*dim_rep + rrep, cf*dim_rep + crep] = mat_to_enter[rrep, crep]                   
    return (dim_rep, dim_fox, ring, '\n', bigmat)

def twisted_homology(pd,matrices):
    # Twisted chain matrices (d1, d2) of pd under a representation (list of matrices).
    foxmat = fox_matrix(pd)
    foxmatrep1 = fox_matrix_rep(foxmat, matrices)[4] 
    foxmat0 = fox_matrix0(pd)
    foxmatrep0 = fox_matrix_rep(foxmat0, matrices)[4] 
    return(foxmatrep0.transpose(), foxmatrep1.transpose())   

def cyclic_branched_cover_homology(pd, n):
    # H_1 of the n-fold cyclic branched cover of pd, via Smith normal form of the twisted Fox matrix.
    cr = len(pd)
    mat = matrix(ZZ, n, n)
    for i in range(n):
        mat[(i+1)%n, i] = 1
    matrices = [mat for i in range(cr)]
    homology_presentation = twisted_homology(pd, matrices)
    smith_f =  homology_presentation[1].smith_form()[0] 
    diag = [ abs(smith_f[i,i])  for i in range(n*(cr -1)) if (abs(smith_f[i,i]) != 1 and abs(smith_f[i,i]) != 0)]
    factors = [ list(factor(x)) for x in diag]
    factors_comb = sorted(flatten(factors,  max_level=1))
    factors_simp = [  x[0]^x[1] for x in factors_comb]
    return factors_simp


# ---------------------------------------------------------------------
# 3. Metacyclic representations of pi_K
# ---------------------------------------------------------------------

def find_p_roots_mod_q(q,p):
    # All elements of (Z/qZ)* with multiplicative order p.
    list = []
    for k in range(2, q):
        order = 0
        power = 1
        ct = 0
        divs = divisors(q)
        notrelprime = flatten([[n*d for n in range(q/d) ] for  d in divs[1:len(divs)-1]])
        if k not in notrelprime:
            while order == 0:
                power = power*k%q
                if power == 1: order = ct
                ct  += 1
            list.append([k, ct])
    return  [x   for x in list if x[1] == p ]

def find_prime_field(d, start = 1):
    # Smallest prime q = 2*n*d+1 with a primitive d-th root of unity in F_q; returns (q, that root).
    n = start
    while not is_prime(2*n*d + 1):
        n +=1
    q = 2*n*d + 1
    F = GF(q)
    gen = F.multiplicative_generator()
    d_gen = (gen^(2*n))%q
    all_gens = sorted([ (d_gen^k)%q  for k in range(d) if gcd(k,d) == 1])
    return q, min(all_gens)

def find_metareps2(pd, d, p, a):
    # Representations of pi_K into M(d,p,a) (t^d=1, r^p=1, trt^-1=r^a), via the kernel of the abelianized relators mod d.
    F = GF(d)
    a = F(a)
    wirt=  wirtinger_presentation(pd)
    mat = matrix(F, len(pd)-1, len(pd), 0)
    for i in range(len(wirt) - 1):
        mat[i,wirt[i][0]] = 1 - a
        mat[i,wirt[i][1]] = a
        mat[i,wirt[i][2]] = -1
    mat_red = mat[0:,1:].transpose()
    K = mat_red.kernel()
    basis_kernel = K.basis()
    return  basis_kernel 

def find_potential_p_metacyclic_targets(pd,p):
    # Candidate (q,a) pairs admitting nontrivial M(q,p,a) representations, from prime factors of the p-fold circulant of the Alexander polynomial.
    k = snappy.Link(pd)
    alex = k.alexander_polynomial()
    Rtemp.<t> = PolynomialRing(ZZ) 
    coeff_list = alex.list() 
    L = [0]*p 
    for i in range(len(coeff_list)):
        L[i%p] = L[ i%p]+ coeff_list[i]
    M = matrix(ZZ, p,p)
    for i in range(p):
        for j in range(p):
            M[i,j] = L[(i + j)%p]
    p_facts = list(factor(abs(det(M))))
    possible_q = [ x[0] for x in p_facts]
    possible_q_a = []
    for q in possible_q:
        poss = find_p_roots_mod_q(q,p)
        possible_q_a.append( [q, [x[0] for x in poss]] )
    all_reps = []
    for trial in possible_q_a:
        q = trial[0]
        for a in trial[1]:
            all_reps.append([[q, p, a], find_metareps2(pd, q, p, a)])
    all_reps_nontrivial = [ x for x in all_reps if len(x[1]) > 0]
    return all_reps_nontrivial

def metacyclic_matrix_rep_prime(Mdpa, Fqr, k):
    # p x p matrix representing t*r^k in M(d,p,a) over F_q.
    d = Mdpa[0]
    p = Mdpa[1]
    a = Mdpa[2]
    q = Fqr[0]
    rt = Fqr[1]
    tmat = matrix(GF(q), p, p, 0)
    rmat = matrix(GF(q), p, p, 0)
    for i in range(p):
        for j in range(p):
            tmat[i%d, (i+1)%p] = 1
    for i in range(p):
            rmat[i,i] = rt^(k*(a^i))%q
    return tmat*rmat

def representation_matrices(Mdpa, Fqr, Fq_rep):
    # Representation matrices for pd's Wirtinger generators under a chosen list of F_q values.
    d = Mdpa[0]
    p = Mdpa[1]
    a = Mdpa[2]
    q = Fqr[0]
    r = Fqr[1]
    matrix_list = [metacyclic_matrix_rep_prime([d,p,a],[q,r],0)]
    for i in range(len(Fq_rep)):
        matrix_list.append( metacyclic_matrix_rep_prime([d,p,a],[q,r],Fq_rep[i]) )
    return matrix_list


# ---------------------------------------------------------------------
# 4. Connected-sum obstruction tests via metacyclic reps (steps 3-5)
# ---------------------------------------------------------------------

def metacyclic_test_diff_primes(pd, n, p1, p2):
    # Step 3-4 obstruction test (d1 != d2): compares the 2nd Betti number of a combined M(p1*p2,n,a) rep to the sum for its M(p1,n,a1)/M(p2,n,a2) summands.
    # Requires p1 != p2, p1 not dividing |H_1(X_n(K2))|, p2 not dividing |H_1(X_n(K1))|. Returns False if obstructed, True if unobstructed.
    FoxMatrix = fox_matrix(pd)
    pot_reps = find_potential_p_metacyclic_targets(pd,n)
    bigprime = find_prime_field(p1*p2)
    prime1 = (bigprime[0], bigprime[1]^p2%(bigprime[0]))
    prime2 = (bigprime[0], bigprime[1]^p1%(bigprime[0]))
    m1 =inverse_mod(p2*p2,p1) 
    m2 =inverse_mod(p1*p1,p2)
    split = [[p1,[]],[p2,[]]]
    for rep in pot_reps:
        if rep[0][0] == p1:
            split[0][1].append(rep)
        if rep[0][0] == p2:
            split[1][1].append(rep)
    for i in range(len(split)):
        for i2 in range(len(split[i][1])):
            for j in range(i+1, len(split)):
                for j2 in range(len(split[j][1])):
                    bigrep = tuple(p2*m1*vector(split[i][1][i2][1][0].change_ring(ZZ))+p1*m2*vector(split[j][1][j2][1][0].change_ring(ZZ)))
                    a1 = split[i][1][i2][0][2]
                    a2 = split[j][1][j2][0][2]
                    newa = (inverse_mod(p2-p1, p1*p2)*(a1*p2-a2*p1))%(p1*p2)
                    rep_mats = representation_matrices([p1*p2, n, newa], bigprime, bigrep)
                    BigMatrix = fox_matrix_rep(FoxMatrix, rep_mats)[4]
                    B= matrix(GF(bigprime[0]), BigMatrix)
                    Ker = B.kernel()
                    Betti = Ker.dimension()
                    rep_mats1 = representation_matrices([p1, n, a1], prime1, split[i][1][i2][1][0])
                    BigMatrix1 = fox_matrix_rep(FoxMatrix, rep_mats1)[4]
                    B1= matrix(GF(prime1[0]), BigMatrix1)
                    Ker1 = B1.kernel()
                    Betti1 = Ker1.dimension()
                    rep_mats2 = representation_matrices([p2, n, a2], prime2, split[j][1][j2][1][0])
                    BigMatrix2 = fox_matrix_rep(FoxMatrix, rep_mats2)[4]
                    B2 = matrix(GF(prime2[0]), BigMatrix2)
                    Ker2 = B2.kernel()
                    Betti2 = Ker2.dimension()
                    if((Betti != Betti1+Betti2) and (Betti != Betti1+Betti2+1)):
                        return False
    return True

def metacyclic_test_same_prime(pd, n, p):
    # Step 5 obstruction test (d1 = d2 = p): compares 2nd Betti numbers across a pencil of M(p,n,a) representations.
    # Requires p | |H_1(X_n(K1))| and p | |H_1(X_n(K2))|, p^2 dividing neither. Returns False if obstructed, True if unobstructed.
    FoxMatrix = fox_matrix(pd)
    pot_reps = find_potential_p_metacyclic_targets(pd,n)
    bigprime = find_prime_field(p)
    pot_reps_p = []
    for rep in pot_reps:
        if rep[0][0] == p:
            pot_reps_p.append(rep)
    coefflist = [[0,1]]
    for i in range(p):
        coefflist.append([1,i])
    for rep_list in pot_reps_p:
        for i in range(len(rep_list[1])):
            for j in range(i+1,len(rep_list[1])):
                proj_pairs = [[a,b] for a,b in zip(rep_list[1][i], rep_list[1][j])]
                proj_reps = [[x*l[0]+y*l[1] for l in proj_pairs] for x,y in coefflist]
                rep_mats = representation_matrices(rep_list[0], bigprime, proj_reps[0])
                BigMatrix = fox_matrix_rep(FoxMatrix, rep_mats)[4]
                B= matrix(GF(bigprime[0]), BigMatrix)
                Ker = B.kernel()                    
                Betti_01 = Ker.dimension()
                rep_mats = representation_matrices(rep_list[0], bigprime, proj_reps[1])
                BigMatrix = fox_matrix_rep(FoxMatrix, rep_mats)[4]
                B= matrix(GF(bigprime[0]), BigMatrix)
                Ker = B.kernel()                    
                Betti_10 = Ker.dimension()
                for k in range(2, len(proj_reps)):
                    rep_mats = representation_matrices(rep_list[0], bigprime, proj_reps[k])
                    BigMatrix = fox_matrix_rep(FoxMatrix, rep_mats)[4]
                    B= matrix(GF(bigprime[0]), BigMatrix)
                    Ker = B.kernel()                    
                    Betti = Ker.dimension()
                    if Betti < Betti_01 + Betti_10:
                        return False #connected sum obstructed
    return True #primeness unconfirmed


# ---------------------------------------------------------------------
# 5. Homology-order utilities (step 2)
# ---------------------------------------------------------------------

def poly_to_order_hom_sqrt(p, n):
    # |H_1(X_n(K))| from HF/Alexander polynomial p; square root taken when n is odd (H_1 splits into two isomorphic summands).
    R1.<s,t> = PolynomialRing(QQ)
    hf_poly = R1(p)
    R2.<t> = PolynomialRing(ZZ) 
    alex_poly = R2(hf_poly.subs(s = -1)) 
    coeff_list = alex_poly.list() 
    L = [0]*n 
    for i in range(len(coeff_list)):
        L[i%n] = L[ i%n]+ coeff_list[i]
    M = matrix(ZZ, n,n)
    for i in range(n):
        for j in range(n):
            M[i,j] = L[(i + j)%n]
    detvalue = abs(det(M))
    if n%2 == 1:
        detvalue = sqrt(detvalue)
    return detvalue

def orders_of_subgroups(L):
    # All possible subgroup orders of a finite abelian group with invariant factors L (products of subsets of L).
    subgroups = all_subsets(L)
    orders = [1]
    for x in subgroups:
        if len(x) > 0:
            pr = 1
            for g in x:
                pr = pr * g
            orders.append(pr)
    orders_unique = []
    for x in orders:
        if x not in orders_unique:
            orders_unique.append(x)
    return sorted(orders_unique)


# ---------------------------------------------------------------------
# 6. Alexander/HF polynomial factorization and symmetry tests (step 1)
# ---------------------------------------------------------------------

def hf_polynomial(pd):
    # Knot Floer homology polynomial Omega_K(s,t) of pd, as (numerator, denominator).
    R.<s,t> = PolynomialRing(QQ)
    k = snappy.Link(pd)
    homology_data = k.knot_floer_homology()
    ranks = homology_data['ranks']
    key_val_list = [[key, value] for key, value in ranks.items()]
    hfpoly = 0
    for pair in key_val_list:
        hfpoly = hfpoly + pair[1]*s^pair[0][1]*t^pair[0][0]
    return [hfpoly.numerator(), hfpoly.denominator()]

def factorization_poly(F):
    # Irreducible factors of F, with multiplicity, as a flat list.
    R.<s,t> = PolynomialRing(QQ)
    F_poly = R(F)
    factors = F_poly.factor()
    factor_list = []
    for factor, exponent in factors:
        factor_list.extend([factor] * exponent)
    return(factor_list)

def all_subsets(L):
    # All subsets of list L.
    if L == []:
        return [[]]
    subsets = []
    first = L[0]
    rest = L[1:]
    for subset in all_subsets(rest):
        subsets.append(subset)
        subsets.append([first] + subset)
    return subsets

def all_factors_poly(F):
    # All divisors of F (products of subsets of its irreducible factors), up to units.
    R.<s,t> = PolynomialRing(QQ)
    F_poly = R(F)
    facts = factorization_poly(F_poly)
    facts_all = all_subsets(facts)
    factors_comb = []
    for i in range(len(facts_all)):
        facts_all_i = facts_all[i]
        v = R(1)
        for j in range(len(facts_all_i)):
            v = v * facts_all_i[j]
        factors_comb.append(v)
    return factors_comb

def sym_test_poly(F):
    # Whether F is positive-symmetric (invariant under the Alexander/HF duality (s,t) -> (1/s,1/t), up to degree shift).
    R.<s,t> = PolynomialRing(QQ)
    F_poly = R(F)
    terms_and_exponents = [[list(exp), coeff] for exp, coeff in F_poly.dict().items()]
    te_sorted = sorted(terms_and_exponents, reverse = True) 
    alex_degree =  max([x[0][1] for x in te_sorted] )
    te_sorted_flipped = [ [ [x[0][0] - 2*x[0][1] + alex_degree, alex_degree - x[0][1]], x[1]] for x in te_sorted]
    te_sorted_flipped = sorted(te_sorted_flipped, reverse = True)     
    return te_sorted == te_sorted_flipped

def all_sym_factorizations(F):
    # Divisors of F that individually pass the positive-symmetric test.
    factors = all_factors_poly(F)
    sym_facts = [ x  for x in factors if sym_test_poly(x) ]
    return list(set(sym_facts))

def all_sym_factor_pairs(F):
    # Step 1 factorizations F = F1*F2 with both factors positive-symmetric, as (F1,F2) pairs.
    R.<s,t> = PolynomialRing(QQ)
    F_poly = R(F)
    all_sym_facts = all_sym_factorizations(F)
    pairs = sorted([   sorted([x, F_poly/x]) for x in all_sym_facts ])
    unique_pairs = []
    for x in pairs:
        if x not in unique_pairs:
            unique_pairs.append(x) 
    return unique_pairs 

def check_pos_coeff(F):
    # Whether F's coefficients are all non-negative or all non-positive.
    R.<s,t> = PolynomialRing(QQ)
    F_poly = R(F)
    coeff_list = list(R(F).dict().values())
    mn = min(coeff_list)
    mx = max(coeff_list)
    if mn*mx > 0:
        return True
    else:
        return False


# ---------------------------------------------------------------------
# 7. Jones and HOMFLY polynomial tests (step 1)
# ---------------------------------------------------------------------

S.<q> = LaurentPolynomialRing(QQ)
P.<v,z> = LaurentPolynomialRing(ZZ)
s,t = symbols("s,t")
def safe_jones(K):
    # K's Jones polynomial; retries since SnapPy's morse-exhaustion routine can raise AssertionError.
    for i in range(10):
        try:
            return K.jones_polynomial()
        except AssertionError:
            print("AssertionError in attempt ", i+1)
    print("AssertionError failure for all 10 attempts.")
    return None

def Jones_test(K, fac):
    # Jones-polynomial obstruction test: drop factorizations in fac whose HF-detected-knot Jones polynomial doesn't divide K's.
    jones = safe_jones(K)
    for i in range(len(hflist)):
        if fac.count(hflist[i]) !=0:
            testlist = [jonesdiv(corjones, jones) for corjones in joneslist[i]]
            orlist = any(testlist)
            if (orlist == False):
                return True # => HF splitting impossible
    return False # => HF splitting still possible

def jonesdiv(p1, p2):
    # Whether Laurent polynomial p1 divides Laurent polynomial p2.
    mindeg1 = p1.degree() - len(list(p1)) + 1
    mindeg2 = p2.degree() - len(list(p2)) + 1
    shift = abs(min(mindeg1, mindeg2))
    l1 = [0 for i in range(shift+mindeg1)] + list(p1)
    l2 = [0 for i in range(shift+mindeg2)] + list(p2)
    l1.reverse()
    l2.reverse()
    quo,r = np.polynomial.polynomial.polydiv(np.array(l2), np.array(l1))
    if list(r) != [0]:
        return False
    return True

def HOMFLY_TEST(pd, fac):
    # HOMFLY-polynomial obstruction test: drop factorizations in fac whose HF-detected-knot HOMFLY polynomial doesn't divide pd's.
    K = snappy.Link(pd)
    Ksage = K.sage_link()
    homfly = Ksage.homfly_polynomial(normalization = 'vz')
    for i in range(len(hflist)):
        if fac.count(hflist[i]) !=0:
            testlist = [homfly.quo_rem(corhomfly)[1] for corhomfly in homflylist[i]]
            test = testlist.count(0)
            if (test == 0):
                return False # => HF splitting impossible, should remove fac from list
    return True # => HF splitting still possible, no obstruction



################# Jones polys, HOMFLY polys, and HF polys of HF-detected knots (in same order) ###############

#Note: Only includes P(-3, 3, 2n+1) for 2n+1 in {1, 3, 5, 7, 9}. All have the same HF poly; Jones and HOMFLY polys are formulaic.

joneslist = [[q^2 + q^6 - q^8, -q^(-8) + q^(-6) + q^(-2)], #T(2,3)
             [q^(-4) - q^(-2) + 1 - q^2 + q^4], #4_1
             [q^4 + q^8 - q^(10) + q^(12) - q^(14), -q^(-14) + q^(-12) - q^(-10) + q^(-8) + q^(-4)], #T(2,5)
             [q^2 - q^4 + 2*q^6 - q^8 + q^(10) - q^(12), q^(-2) - q^(-4) + 2*q^(-6) - q^(-8) + q^(-10) - q^(-12)], #5_2
             [q^(-4) - q^(-2) + 1 - q^(10) + q^(12) - q^(14) + 2*q^(16) - q^(18), q^(4) - q^(2) + 1 - q^(-10) + q^(-12) - q^(-14) + 2*q^(-16) - q^(-18)], #Wh^+(T(2,3),2)
             [q^8-q^6+q^4-2*q^2+2-q^(-2)+q^(-4), q^(-8)-q^(-6)+q^(-4)-2*q^(-2)+2-q^(2)+q^(4), #P(-3,3,1)
             q^(12) - q^(10) +q^8 -2*q^6 +q^4 -q^2+2, q^(-12) -q^(-10) +q^(-8) -2*q^(-6) +q^(-4) -q^(-2)+2, #P(-3,3,3)
             q^(16) -q^(14) +q^(12) -2*q^(10) +q^8 -q^6 +q^4 +1, q^(-16) -q^(-14) +q^(-12) -2*q^(-10) +q^(-8) -q^(-6) +q^(-4) +1, #P(-3,3,5)
             q^(20) -q^(18) +q^(16) -2*q^(14) +q^(12) -q^(10) +q^8 +1, q^(-20) -q^(-18) +q^(-16) -2*q^(-14) +q^(-12) -q^(-10) +q^(-8) +1, #P(-3,3,7)
             q^(24) -q^(22) +q^(20) -2*q^(18) +q^(16) -q^(14) +q^(12) +1, q^(-24) -q^(-22) +q^(-20) -2*q^(-18) +q^(-16) -q^(-14) +q^(-12) +1, #P(-3,3,9)
             q^(28) -q^(26) +q^(24) -2*q^(22) +q^(20) -q^(18) +q^(16) +1, q^(-28) -q^(-26) +q^(-24) -2*q^(-22) +q^(-20) -q^(-18) +q^(-16) +1], #P(-3,3,11)
             [-q^(-16) + q^(-14) - q^(-12) + q^(-10) + q^(-6) - q^4 + q^6, -q^(16) + q^(14) - q^(12) + q^(10) + q^(6) - q^(-4) + q^(-6), #15n43522
             -q^(-8) + q^(-6) + 1 + q^6 - q^8 + q^10 - 2*q^12 + q^14, -q^(8) + q^(6) + 1 + q^(-6) - q^(-8) + q^(-10) - 2*q^(-12) + q^(-14)]] #Wh^-(T(2,3),2)

homflylist = [[-v^4 + v^2*z^2 + 2*v^2, v^-2*z^2 + 2*v^-2 - v^-4], #T(2,3)
 [v^2 - z^2 - 1 + v^-2, v^2 - z^2 - 1 + v^-2], #4_1
 [-v^6*z^2 + v^4*z^4 - 2*v^6 + 4*v^4*z^2 + 3*v^4,  v^-4*z^4 + 4*v^-4*z^2 + 3*v^-4 - v^-6*z^2 - 2*v^-6], #T(2,5)
 [-v^6 + v^4*z^2 + v^4 + v^2*z^2 + v^2,  v^-2*z^2 + v^-2 + v^-4*z^2 + v^-4 - v^-6], #5_2
 [-v^8*z^2 + v^4*z^6 - v^6*z^2 + 6*v^4*z^4 - v^2*z^6 - v^6 + 10*v^4*z^2 - 7*v^2*z^4 + 5*v^4 - 14*v^2*z^2 + z^4 - 7*v^2 + 4*z^2 + 4, z^4 - v^-2*z^6 + 4*z^2 - 7*v^-2*z^4 + v^-4*z^6 + 4 - 14*v^-2*z^2 + 6*v^-4*z^4 - 7*v^-2 + 10*v^-4*z^2 + 5*v^-4 - v^-6*z^2 - v^-6 - v^-8*z^2], #Wh^+(T(2,3),2)
 [v^4 - v^2*z^2 - v^2 - z^2 + v^-2, v^2 - z^2 - v^-2*z^2 - v^-2 + v^-4, #P(-3,3,1)
 v^6 - v^4*z^2 - v^4 - v^2*z^2 - v^2 + 2,  2 - v^-2*z^2 - v^-2 - v^-4*z^2 - v^-4 + v^-6, #P(-3,3,3)
 v^8 - v^6*z^2 - v^6 - v^4*z^2 - v^4 + v^2 + 1,  1 + v^-2 - v^-4*z^2 - v^-4 - v^-6*z^2 - v^-6 + v^-8, #P(-3,3,5)
 v^10 - v^8*z^2 - v^8 - v^6*z^2 - v^6 + v^4 + 1, 1 + v^-4 - v^-6*z^2 - v^-6 - v^-8*z^2 - v^-8 + v^-10, #P(-3,3,7)
 v^12 - v^10*z^2 - v^10 - v^8*z^2 - v^8 + v^6 + 1, 1 + v^-6 - v^-8*z^2 - v^-8 - v^-10*z^2 - v^-10 + v^-12, #P(-3,3,9)
 v^14 - v^12*z^2 - v^12 - v^10*z^2 - v^10 + v^8 + 1, 1 + v^-8 - v^-10*z^2 - v^-10 - v^-12*z^2 - v^-12 + v^-14], #P(-3,3,11)
 [z^2 - v^-2*z^4 + 1 - 2*v^-2*z^2 - v^-2 + v^-4*z^2 + v^-6*z^4 + 4*v^-6*z^2 + 3*v^-6 - v^-8*z^2 - 2*v^-8, -v^8*z^2 + v^6*z^4 - 2*v^8 + 4*v^6*z^2 + 3*v^6 + v^4*z^2 - v^2*z^4 - 2*v^2*z^2 - v^2 + z^2 + 1, #15n43522
 -v^2*z^4 + z^6 - 4*v^2*z^2 + 7*z^4 - v^-2*z^6 - 3*v^2 + 14*z^2 - 6*v^-2*z^4 + 8 - 10*v^-2*z^2 - 5*v^-2 + v^-4*z^2 + v^-4 + v^-6*z^2,  v^6*z^2 - v^2*z^6 + v^4*z^2 - 6*v^2*z^4 + z^6 + v^4 - 10*v^2*z^2 + 7*z^4 - 5*v^2 + 14*z^2 - v^-2*z^4 + 8 - 4*v^-2*z^2 - 3*v^-2]] #Wh^-(T(2,3),2)

hflist = [s^2*t^2 + s*t + 1, #T(2,3)
          s^2*t^2 + 3*s*t + 1, #4_1
          s^4*t^4 + s^3*t^3 + s^2*t^2 + s*t + 1, #T(2,5)
          2*s^2*t^2 + 3*s*t + 2, #5_2
          s^3*t + 2*s^2*t^2 + 4*s*t + 2, #Wh^+(T(2,3),2)
          2*s^2*t^2 + 5*s*t + 2, #P(-3,3,2n+1)
          2*s^2*t^2 + s^2*t + 4*s*t + 2] #15n43522 and Wh^-(T(2,3),2)


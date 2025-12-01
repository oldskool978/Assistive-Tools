import mpmath
from mpmath import mp
import os
import re
import argparse
import json
import glob
import datetime

# --- 100-DPS MAVERICK STANDARD ---
# We calculate to 100 decimal places to eliminate floating point drift at the source.
mp.dps = 100

DATA_MAP = {}

TARGET_MANIFEST = {
    "KAISER_13", "FARID_P", "FARID_D", "ENTROPY_LUT", "LINEAR_LUT_HQ",
    "KURTOSIS_LUT", "SIGMA_LUT", "TURBO", "INFERNO",
    "BAYER_LATTICE", "M1", "M2", "PHYSICS", "LUMA_COEFFS",
    "HVS_MASK_LUT",
    "MS_SSIM_WEIGHTS", "SWD_PROJECTIONS", "PYRAMID_KERNEL"
}

def get_cache_filename(): return f"{datetime.date.today().isoformat()}_manifold.IGD"

def clean_cache():
    for f in glob.glob("*_manifold.IGD"):
        try: os.remove(f); print(f"[System] Cache purged: {f}")
        except: pass

def clean_target(filename):
    if not os.path.exists(filename): return
    with open(filename, 'r', encoding='utf-8') as f: content = f.read()
    pattern = re.compile(r"(// <INJECT_(\w+)>)([\s\S]*?)(// </INJECT_\2>)", re.MULTILINE)
    def reset_block(m): return f"{m.group(1)}\n            const {m.group(2)} = null;\n            {m.group(4)}"
    new_content = pattern.sub(reset_block, content)
    if new_content != content:
        with open(filename, 'w', encoding='utf-8') as f: f.write(new_content)
        print(f"[System] Target reset: {filename}")

def load_cache():
    files = glob.glob("*_manifold.IGD")
    if not files: return
    files.sort(key=os.path.getmtime, reverse=True)
    try:
        with open(files[0], 'r', encoding='utf-8') as f: DATA_MAP.update(json.load(f))
        print(f"[System] Data loaded from {files[0]}")
    except: pass

def prune_cache():
    for k in list(DATA_MAP.keys()):
        if k not in TARGET_MANIFEST: del DATA_MAP[k]

def save_cache():
    try:
        with open(get_cache_filename(), 'w', encoding='utf-8') as f: json.dump(DATA_MAP, f, indent=0)
        print("[System] Data serialized.")
    except: pass

def register_injection(name, data):
    DATA_MAP[name] = {'data': [mp.nstr(x, 30) for x in data]}
    print(f"   [+] Computed {name} ({len(data)} elements)")

def format_js_array(name, entry):
    data = entry['data']
    rows = ["            " + ", ".join(data[i:i+8]) for i in range(0, len(data), 8)]
    return f"\n            // INJECTED: {name} ({len(data)} vals)\n            const {name} = new Float64Array([\n{',\n'.join(rows)}\n            ]);"

def safe_findroot(func, low, high):
    try:
        f_low = func(low); f_high = func(high)
        if f_low * f_high > 0:
            found = False
            for i in range(1, 20):
                mid = low * (10**i)
                if mid >= high: break
                if func(low) * func(mid) < 0: high = mid; found = True; break
                low = mid
            if not found: return (low + high) / 2
        return mp.findroot(func, (low, high), solver='bisect', tol=mp.mpf('1e-25'))
    except: return (low + high) / 2

def ensure_robust_metrics():
    # [NEW] Tri-Fold NPS Kernels
    missing = [k for k in ["MS_SSIM_WEIGHTS", "SWD_PROJECTIONS", "PYRAMID_KERNEL"] if k not in DATA_MAP]
    if not missing: return
    print(f"\n[Kernel] Calculating Robust NPS Matrices: {missing}")

    if "MS_SSIM_WEIGHTS" in missing:
        # Wang et al., 2003: Empirical weights for 5 scales
        # We normalize strictly to sum to 1.0 at 100-dps
        raw_w = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        total = sum([mp.mpf(str(x)) for x in raw_w])
        register_injection("MS_SSIM_WEIGHTS", [mp.mpf(str(x))/total for x in raw_w])

    if "SWD_PROJECTIONS" in missing:
        # Superior Deterministic Slicing (Riemann Sum of Radon Transform)
        # Uniform distribution on Unit Circle > Random Monte Carlo
        print("   ... Generating Sliced Wasserstein Vectors (Uniform)")
        slices = 64 # High fidelity angular resolution
        projections = []
        step = mp.pi / mp.mpf(slices)
        for i in range(slices):
            theta = i * step
            projections.append(mp.cos(theta))
            projections.append(mp.sin(theta))
        register_injection("SWD_PROJECTIONS", projections)

    if "PYRAMID_KERNEL" in missing:
        # Burt & Adelson: a = 0.375 is the point of Zero Aliasing contribution
        print("   ... Deriving Laplacian Pyramid Kernel (a=0.375)")
        a = mp.mpf('0.375')
        k0 = mp.mpf('0.25') - a/2
        k1 = mp.mpf('0.25')
        k2 = a
        # Symmetric 5-tap: [k0, k1, k2, k1, k0]
        register_injection("PYRAMID_KERNEL", [k0, k1, k2, k1, k0])

def ensure_math():
    missing = [k for k in TARGET_MANIFEST if k not in DATA_MAP]
    if not missing: return
    print(f"\n[Kernel] Calculating: {missing}")
    
    if "PHYSICS" in missing:
        # 1. Robust Sigma Scale (Inverse Error Function)
        # Scale factor to convert MAD to StDev for Normal Distribution
        sigma_scale = mp.mpf(1) / (mp.sqrt(2) * mp.erfinv(0.5))
        
        # 2. SSIM Stability Constants (Dynamic Range Equations)
        # L = 1.0 (Normalized Space). C1 = (K1*L)^2, C2 = (K2*L)^2
        k1 = mp.mpf('0.01')
        k2 = mp.mpf('0.03')
        L = mp.mpf('1.0')
        ssim_c1 = (k1 * L) ** 2
        ssim_c2 = (k2 * L) ** 2
        
        # 3. Quantization Step (8-bit reference)
        quant_step = mp.mpf(1) / (mp.power(2, 8) - 1)
        
        # 4. Maximum Entropy (Uniform distribution over 25 bins)
        # H_max = log2(N_bins)
        max_entropy = mp.log(25) / mp.log(2)
        
        register_injection("PHYSICS", [sigma_scale, ssim_c1, ssim_c2, quant_step, max_entropy])

    if "LUMA_COEFFS" in missing:
        # Rec. 709 (HDTV Standard)
        register_injection("LUMA_COEFFS", [mp.mpf('0.2126390059'), mp.mpf('0.7151686788'), mp.mpf('0.0721923154')])

    if "M1" in missing:
        # Oklab M1 (D65 white point to LMS) - Björn Ottosson
        register_injection("M1", [mp.mpf(x) for x in ["0.4122214708", "0.5363325363", "0.0514459929", "0.2119034982", "0.6806995451", "0.1073969566", "0.0883024619", "0.2817188376", "0.6299787005"]])
    if "M2" in missing:
        # Oklab M2 (LMS to Lab) - Björn Ottosson
        register_injection("M2", [mp.mpf(x) for x in ["0.2104542553", "0.7936177850", "-0.0040720468", "1.9779984951", "-2.4285922050", "0.4505937099", "0.0259040371", "0.7827717662", "-0.8086757660"]])

    if "BAYER_LATTICE" in missing:
        # Recursive definition of Bayer Matrix (Ordered Dithering)
        def b_rec(n):
            if n==0: return mp.matrix([[0]])
            p=b_rec(n-1); s=2**(n-1); m=mp.matrix(2*s,2*s)
            for r in range(s):
                for c in range(s):
                    v=p[r,c]*4; m[r,c]=v; m[r+s,c+s]=v+1; m[r,c+s]=v+2; m[r+s,c]=v+3
            return m
        b8=b_rec(3); flat=[(b8[r,c]/64.0)-0.5 for r in range(8) for c in range(8)]
        register_injection("BAYER_LATTICE", flat)

    if "KURTOSIS_LUT" in missing or "SIGMA_LUT" in missing:
        print("   ... Solving Generalized Gaussian Distributions (Newton-Raphson/Bisect)")
        # Range covers Laplacian (3.0) to Uniform (1.8) to Impulsive (>3.0)
        k_min = mp.mpf('1.805'); k_max = mp.mpf('10.0'); kurt_lut = []; sigma_lut = []
        std_mad_gauss = mp.sqrt(2) * mp.erfinv(0.5)
        # Relationship between Beta (shape) and Kurtosis
        ggd_k = lambda b: (mp.gamma(5/b)*mp.gamma(1/b))/(mp.gamma(3/b)**2)
        
        last_beta = mp.mpf('50.0') 
        
        for i in range(256):
            tk = k_min + (mp.mpf(i)/255)*(k_max-k_min)
            # Singularity check near Gaussian (Beta=2, Kurtosis=3)
            if abs(tk - 3.0) < 0.01: beta = mp.mpf(2.0)
            else:
                upper_bound = max(last_beta + 0.1, mp.mpf('5.0')) if i == 0 else last_beta + 0.1
                beta = safe_findroot(lambda b: mp.re(ggd_k(b)) - tk, mp.mpf('0.02'), upper_bound)
            
            last_beta = beta 
            kurt_lut.append(beta)
            shape = 1/beta
            # Solve for Median (CDF = 0.5)
            med_pow = safe_findroot(lambda t: mp.gammainc(shape, 0, t, regularized=True) - 0.5, shape*mp.mpf('0.0001'), shape*mp.mpf('10.0'))
            theoretical_mad = (med_pow ** (1/beta))
            theoretical_std = mp.sqrt(mp.gamma(3/beta)/mp.gamma(1/beta))
            sigma_lut.append((theoretical_std / theoretical_mad) / std_mad_gauss)
        register_injection("KURTOSIS_LUT", kurt_lut); register_injection("SIGMA_LUT", sigma_lut)

    if "KAISER_13" in missing:
        # Modified Bessel Function of the First Kind (I0) for Windowing
        beta=mp.mpf(4.0); center=6.0; den=mp.besseli(0,beta)
        # Integrate over pixel width for maximal accuracy
        half_k = [mp.quad(lambda x: mp.besseli(0,beta*mp.sqrt(1-((x-center)/6)**2))/den if abs((x-center)/6)<=1 else 0,[i-0.5,i+0.5]) for i in range(7)]
        k_data = half_k + half_k[-2::-1]
        tot=sum(k_data); register_injection("KAISER_13", [x/tot for x in k_data])

    if "LINEAR_LUT_HQ" in missing:
        # sRGB Inverse Gamma (IEC 61966-2-1:1999)
        lin=[]; a=mp.mpf('0.055'); g=mp.mpf('2.4')
        inv_max = mp.mpf(1) / 4095
        for i in range(4096): 
            v = i * inv_max
            lin.append(v/mp.mpf('12.92') if v<=mp.mpf('0.04045') else ((v+a)/mp.mpf('1.055'))**g)
        lin.append(lin[-1]) 
        register_injection("LINEAR_LUT_HQ", lin)

    if "FARID_P" in missing:
        # Optimal 5-tap Interpolator/Differentiator
        p=[mp.mpf(x) for x in ["0.0376592045236901","0.249153396177344","0.426374798597931"]]
        d=[mp.mpf(x) for x in ["0.109603762960254","0.276690988455557"]]
        register_injection("FARID_P", [p[0],p[1],p[2],p[1],p[0]]); register_injection("FARID_D", [d[0],d[1],0,-d[1],-d[0]])

    if "ENTROPY_LUT" in missing:
        # Shannon Entropy log base 2
        ent=[0]+[-p*(mp.log(p)/mp.log(2)) for p in [mp.mpf(i)/25.0 for i in range(1,26)]]
        register_injection("ENTROPY_LUT", ent)

    if "HVS_MASK_LUT" in missing:
        print("   ... Deriving HVS Masking Curve (Naka-Rushton Equation)")
        q_step = mp.mpf(1) / mp.mpf(255)
        sigma_hvs = q_step * 10.0 # Semisaturation constant
        n_hvs = mp.mpf('2.2')     # Exponent
        lut_size = 4096
        mask_curve = []
        inv_lut_max = mp.mpf(1) / (lut_size - 1)
        for i in range(lut_size):
            g = i * inv_lut_max
            if g == 0: w = mp.mpf(1.0)
            else:
                denom = 1 + (g / sigma_hvs) ** n_hvs
                w = 1 / denom
            mask_curve.append(w)
        mask_curve.append(mask_curve[-1]) 
        register_injection("HVS_MASK_LUT", mask_curve)

    if "TURBO" in missing:
        # Google AI Turbo Colormap (Polynomial Approximation)
        C=[[0.13572138,4.61539260,-42.66032258,132.13108234,-152.94239396,59.28637943],[0.09140261,2.19418839,4.84296658,-14.18503333,4.27729857,2.82956604],[0.10667330,12.64194608,-60.58204836,110.36276771,-89.90360919,27.34824973]]
        t_d=[]
        inv_max = mp.mpf(1) / 4095
        for i in range(4096):
            x = i * inv_max
            rgb=[]
            for k in C:
                r=mp.mpf(str(k[-1]))
                for c in reversed(k[:-1]): r=r*x+mp.mpf(str(c))
                rgb.append(max(0,min(1,r)))
            t_d.extend(rgb)
        register_injection("TURBO", t_d)

    if "INFERNO" in missing:
        # Matplotlib Inferno (Cosine Approximation)
        ctrl=[[0.001462,0.000466,0.013866],[0.039608,0.031090,0.133515],[0.106368,0.028426,0.275762],[0.216423,0.031167,0.392770],[0.340087,0.060082,0.446213],[0.477040,0.115871,0.429507],[0.612902,0.202487,0.355196],[0.738931,0.314778,0.241356],[0.850714,0.457647,0.106581],[0.938815,0.637251,0.048689],[0.987053,0.834970,0.168624],[0.988362,0.998364,0.644924]]
        i_d=[]
        inv_max = mp.mpf(1) / 4095
        for i in range(4096):
            t = i * inv_max 
            idx=t*(len(ctrl)-1); i0=int(idx); i1=min(i0+1,len(ctrl)-1); f=(1-mp.cos((idx-i0)*mp.pi))/2
            i_d.extend([mp.mpf(str(ctrl[i0][j]))+(mp.mpf(str(ctrl[i1][j]))-mp.mpf(str(ctrl[i0][j])))*f for j in range(3)])
        register_injection("INFERNO", i_d)

def process_file(filename):
    if not os.path.exists(filename): return
    with open(filename, 'r', encoding='utf-8') as f: content = f.read()
    pattern = re.compile(r"(// <INJECT_(\w+)>)([\s\S]*?)(// </INJECT_\2>)", re.MULTILINE)
    def rep(m): return f"{m.group(1)}{format_js_array(name, DATA_MAP[name])}\n            {m.group(4)}" if (name:=m.group(2)) in DATA_MAP else m.group(0)
    with open(filename, 'w', encoding='utf-8') as f: f.write(pattern.sub(rep, content))
    print(f"[Inject] Updated {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--clean", action="store_true"); args = parser.parse_args()
    if args.clean: clean_cache(); clean_target("index.html")
    load_cache(); prune_cache(); ensure_math(); ensure_robust_metrics(); save_cache(); process_file("index.html")

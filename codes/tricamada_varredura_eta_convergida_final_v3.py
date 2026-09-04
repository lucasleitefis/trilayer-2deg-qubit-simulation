# coding: utf-8

"""

SIMULACAO DE TRICAMADA: QUBIT EFETIVO, MATRIZ DENSIDADE E TOMOGRAFIA

Autor: Lucas Bez

Versao: publicacao

Objetivo

--------

Executa a evolucao temporal do qubit efetivo de camada em uma tricamada,

calcula a matriz densidade reduzida do subespaco logico {|1>, |3>} e gera

figuras com formatacao profissional para artigo:

1. Figura composta de dinamica: populacoes, coerencia, pureza e vetor de Bloch.

2. Figura composta de tomografia: matrizes densidade ao lado das respectivas esferas de Bloch.

3. Dados numericos e captions em LaTeX.

4. Dados numericos e captions em LaTeX.

Notas

-----

- As figuras principais sao salvas em PDF vetorial.

- Os graficos evitam titulos internos longos; a explicacao fica nas captions.

- Os painéis sao identificados por (a), (b), ... no estilo de artigo.

- O Qiskit e opcional. Se nao estiver instalado, o codigo continua normalmente.

"""

from __future__ import annotations

import os

import time

from dataclasses import dataclass

from typing import Dict, List, Tuple

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image, ImageChops

from matplotlib.colors import Normalize

from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from numpy.fft import fft2, ifft2, fftshift, ifftshift

# -----------------------------------------------------------------------------

# QISKIT OPCIONAL

# -----------------------------------------------------------------------------

try:

    from qiskit.visualization import plot_bloch_vector

    QISKIT_AVAILABLE = True

    QISKIT_IMPORT_ERROR = None

except Exception as exc:  # pragma: no cover

    QISKIT_AVAILABLE = False

    QISKIT_IMPORT_ERROR = exc



# -----------------------------------------------------------------------------

# CONFIGURACOES GERAIS

# -----------------------------------------------------------------------------

@dataclass(frozen=True)

class PhysicalParameters:

    hbar_ev: float = 6.58211928e-16

    e_charge: float = 1.60217662e-19

    m_el: float = 9.1093837e-31

    m_eff_factor: float = 0.067

    Delta: float = 0.005        # eV

    U: float = 0.10             # eV

    use_potential: bool = True

    V0: float = 0.003           # eV

    x_barrier: float = 40.0     # nm

    width: float = 10.0         # nm



@dataclass(frozen=True)

class NumericalParameters:
    nx: int = 384
    ny: int = 384
    WX: float = 1200.0          # nm
    WY: float = 1200.0          # nm
    dt_fs: float = 0.5          # fs

    nt: int = 15000

    save_step: int = 50

    sigma: float = 15.0         # nm

    # Se True, mostra as figuras na tela com plt.show() apos salvar.

    # Se estiver rodando em servidor/headless, mude para False.

    show_figures: bool = False

    # Configuracao da varredura adimensional V0/(2J).

    # O valor 6 reproduz o caso original: V0 = 3 meV para 2J = 0.5 meV.

    sweep_ratios: Tuple[float, ...] = (

        0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0

    )

    # A varredura usa uma janela de \~6 ps, suficiente para conter t_X e

    # o primeiro maximo de transferencia. Para tornar a varredura viavel

    # em computador comum, ela usa dt=2 fs; a simulacao principal permanece

    # com dt=1 fs e nt=15000.
    sweep_dt_fs: float = 0.5
    sweep_nt: int = 9000



@dataclass(frozen=True)

class FigureStyle:

    # Paleta alinhada à figura de populações da tricamada completa.

    pop_1: str = "blue"

    pop_3: str = "red"

    theory: str = "#2f2f2f"

    coherence: str = "forestgreen"

    purity: str = "black"

    purity_theory: str = "gray"

    bloch_x: str = "blue"

    bloch_y: str = "red"

    bloch_z: str = "#4d4d4d"

    # Escala editorial compatível com Fig. pop2/zoom.

    lw_main: float = 1.7

    lw_theory: float = 1.25

    panel_fontsize: int = 18

    label_fontsize: int = 19

    tick_fontsize: int = 15

    legend_fontsize: float = 14.0

    title_fontsize: float = 18.0

    tomo_tick_fontsize: float = 7.0

    tomo_row_fontsize: float = 18.0



def configure_matplotlib() -> None:

    """Define padrão visual unificado, igual ao código das figuras pop2/zoom."""

    plt.rcParams.update({

        "text.usetex": True,

        "font.family": "serif",

        "font.serif": ["Computer Modern Roman"],

        "text.latex.preamble": r"**\u**sepackage{amsmath}",

        "text.color": "black",

        "axes.labelcolor": "black",

        "xtick.color": "black",

        "ytick.color": "black",

        "axes.edgecolor": "black",

        "font.size": 18,

        "axes.labelsize": 19,

        "axes.titlesize": 18,

        "legend.fontsize": 14,

        "xtick.labelsize": 15,

        "ytick.labelsize": 15,

        "axes.linewidth": 0.75,

        "xtick.major.size": 5,

        "ytick.major.size": 5,

        "xtick.major.width": 0.75,

        "ytick.major.width": 0.75,

        "xtick.minor.visible": True,

        "ytick.minor.visible": True,

        "xtick.minor.size": 2.5,

        "ytick.minor.size": 2.5,

        "xtick.minor.width": 0.55,

        "ytick.minor.width": 0.55,

        "xtick.direction": "in",

        "ytick.direction": "in",

        "xtick.top": True,

        "ytick.right": True,

        "legend.frameon": False,

        "pdf.fonttype": 42,

        "ps.fonttype": 42,

    })

def panel_label(ax, label: str, x: float = 0.03, y: float = 0.94) -> None:

    ax.text(

        x, y, label,

        transform=ax.transAxes,

        ha="left", va="top",

        fontsize=18,

        fontweight="normal",

        color="black",

    )



def polish_axes(ax, grid: bool = True) -> None:

    for spine in ax.spines.values():

        spine.set_linewidth(0.75)

    ax.tick_params(which="major", direction="in", length=5, width=0.75)

    ax.tick_params(which="minor", direction="in", length=2.5, width=0.55)

    if grid:

        ax.grid(True, alpha=0.23, linewidth=0.55)



def prepare_axes_for_export(ax) -> None:

    ax.xaxis.label.set_clip_on(False)

    ax.yaxis.label.set_clip_on(False)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():

        tick.set_clip_on(False)



def save_publication_figure(fig, filename: str, png_dpi: int = 450) -> None:

    fig.canvas.draw()

    fig.savefig(filename, format="pdf", dpi=300, transparent=True, bbox_inches=None, pad_inches=0.0)

    fig.savefig(filename.replace(".pdf", ".png"), format="png", dpi=png_dpi, transparent=True, bbox_inches=None, pad_inches=0.0)



# -----------------------------------------------------------------------------

# FUNCOES NUMERICAS

# -----------------------------------------------------------------------------

def fft2_opt(p: np.ndarray) -> np.ndarray:

    return fftshift(fft2(ifftshift(p)))



def ifft2_opt(p: np.ndarray) -> np.ndarray:

    return fftshift(ifft2(ifftshift(p)))



def reduced_density_matrix(

    psi1_field: np.ndarray,

    psi3_field: np.ndarray,

    dx: float,

    dy: float,

) -> Tuple[np.ndarray, float, np.ndarray]:

    """Retorna rho_red, pureza e vetor de Bloch do qubit efetivo de camada."""

    rho11 = np.sum(np.abs(psi1_field) ** 2) * dx * dy

    rho33 = np.sum(np.abs(psi3_field) ** 2) * dx * dy

    rho13 = np.sum(psi1_field * np.conjugate(psi3_field)) * dx * dy

    rho = np.array(

        [[rho11, rho13], [np.conjugate(rho13), rho33]],

        dtype=complex,

    )

    purity = float(np.real(np.trace(rho @ rho)))

    rx = 2.0 * np.real(rho13)

    ry = -2.0 * np.imag(rho13)

    rz = rho11 - rho33

    return rho, purity, np.array([rx, ry, rz], dtype=float)



def run_simulation(

    phys: PhysicalParameters,

    num: NumericalParameters,

    outdir: str,

) -> Dict[str, np.ndarray | List[np.ndarray]]:

    """Executa a evolucao por split-operator no subespaco efetivo {|1>, |3>}."""

    os.makedirs(outdir, exist_ok=True)

    m_eff = phys.m_eff_factor * phys.m_el

    hbar2_2m_ev_nm2 = (phys.hbar_ev * 1e9) ** 2 / (2 * m_eff * 1e-30 / phys.e_charge)

    J = phys.Delta ** 2 / phys.U

    Omega_eff_ev = 2.0 * J

    T_eff_fs = (2.0 * np.pi * phys.hbar_ev / Omega_eff_ev) * 1e15

    tX_fs = T_eff_fs / 2.0

    dx = num.WX / num.nx

    dy = num.WY / num.ny

    x = np.linspace(-num.WX / 2.0, num.WX / 2.0, num.nx)

    y = np.linspace(-num.WY / 2.0, num.WY / 2.0, num.ny)

    X, Y = np.meshgrid(x, y, indexing="ij")

    kx = np.fft.fftfreq(num.nx, d=dx) * 2.0 * np.pi

    ky = np.fft.fftfreq(num.ny, d=dy) * 2.0 * np.pi

    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    K2 = KX ** 2 + KY ** 2

    if phys.use_potential:

        V = phys.V0 * np.exp(-((X - phys.x_barrier) ** 2) / (2.0 * phys.width ** 2))

    else:

        V = np.zeros((num.nx, num.ny))

    cdt = num.dt_fs * 1e-15

    dt_hbar = cdt / phys.hbar_ev

    T_half = np.exp(-1j * (hbar2_2m_ev_nm2 * K2) * (dt_hbar / 2.0))

    U00 = np.zeros((num.nx, num.ny), dtype=np.complex128)

    U01 = np.zeros((num.nx, num.ny), dtype=np.complex128)

    U10 = np.zeros((num.nx, num.ny), dtype=np.complex128)

    U11 = np.zeros((num.nx, num.ny), dtype=np.complex128)

    print("\n" + "=" * 80)

    print("RELATORIO INICIAL")

    print("=" * 80)

    print(f"Qiskit disponivel: {QISKIT_AVAILABLE}")

    if not QISKIT_AVAILABLE:

        print(f"Aviso: Qiskit nao foi importado: {QISKIT_IMPORT_ERROR}")

    print(f"Delta = {phys.Delta * 1000:.3f} meV")

    print(f"U = {phys.U * 1000:.3f} meV")

    print(f"J = Delta^2/U = {J * 1000:.6f} meV")

    print(f"U/Delta = {phys.U / phys.Delta:.1f}")

    print(f"Omega_eff = 2J = {Omega_eff_ev * 1000:.6f} meV")

    print(f"T_eff = {T_eff_fs:.2f} fs | t_X = {tX_fs:.2f} fs")

    print(f"Grade = {num.nx} x {num.ny} | dt = {num.dt_fs:.3f} fs | nt = {num.nt}")

    print(f"Mostrar figuras com plt.show(): {num.show_figures}")

    print(f"Pasta de saida: {os.path.abspath(outdir)}")

    print("=" * 80)

    print("\nPre-calculando operador efetivo...")

    for i in range(num.nx):

        for j in range(num.ny):

            v = V[i, j]

            v_half = v / 2.0

            Omega = np.sqrt(v_half ** 2 + J ** 2)

            if Omega > 1e-14:

                cos_O = np.cos(Omega * dt_hbar)

                sin_O = np.sin(Omega * dt_hbar)

                phase = np.exp(-1j * v_half * dt_hbar)

                U00[i, j] = phase * (cos_O - 1j * (v_half / Omega) * sin_O)

                U01[i, j] = phase * (-1j * (J / Omega) * sin_O)

                U10[i, j] = phase * (-1j * (J / Omega) * sin_O)

                U11[i, j] = phase * (cos_O + 1j * (v_half / Omega) * sin_O)

            else:

                U00[i, j] = 1.0

                U11[i, j] = 1.0

    psi1 = np.exp(-(X ** 2 + Y ** 2) / (2.0 * num.sigma ** 2)).astype(np.complex128)

    psi1 /= np.sqrt(np.sum(np.abs(psi1) ** 2) * dx * dy)

    psi3 = np.zeros((num.nx, num.ny), dtype=np.complex128)

    history: Dict[str, list] = {

        "t": [], "p1": [], "p3": [], "rho13_re": [], "rho13_im": [],

        "coherence": [], "purity": [], "rx": [], "ry": [], "rz": [], "rho": []

    }

    print("\nIniciando simulacao...")

    start = time.time()

    for it in range(num.nt + 1):

        if it % num.save_step == 0:

            rho, purity, bloch = reduced_density_matrix(psi1, psi3, dx, dy)

            history["t"].append(it * num.dt_fs)

            history["p1"].append(float(np.real(rho[0, 0])))

            history["p3"].append(float(np.real(rho[1, 1])))

            history["rho13_re"].append(float(np.real(rho[0, 1])))

            history["rho13_im"].append(float(np.imag(rho[0, 1])))

            history["coherence"].append(float(np.abs(rho[0, 1])))

            history["purity"].append(purity)

            history["rx"].append(float(bloch[0]))

            history["ry"].append(float(bloch[1]))

            history["rz"].append(float(bloch[2]))

            history["rho"].append(rho.copy())

            if it % (20 * num.save_step) == 0:

                print(

                    f"it={it:6d} | t={it * num.dt_fs:9.1f} fs | "

                    f"P1={np.real(rho[0, 0]):.6f} | P3={np.real(rho[1, 1]):.6f} | "

                    f"|rho13|={np.abs(rho[0, 1]):.6f} | Tr(rho^2)={purity:.6f}"

                )

        if it == num.nt:

            break

        # Meio passo cinetico

        psi1 = ifft2_opt(fft2_opt(psi1) * T_half)

        psi3 = ifft2_opt(fft2_opt(psi3) * T_half)

        # Passo de acoplamento + potencial

        psi1_new = U00 * psi1 + U01 * psi3

        psi3_new = U10 * psi1 + U11 * psi3

        psi1, psi3 = psi1_new, psi3_new

        # Meio passo cinetico

        psi1 = ifft2_opt(fft2_opt(psi1) * T_half)

        psi3 = ifft2_opt(fft2_opt(psi3) * T_half)

    elapsed = time.time() - start

    print(f"\nSimulacao concluida em {elapsed:.2f} s")

    # Converter listas numericas para arrays, preservando rho como lista.

    result: Dict[str, np.ndarray | List[np.ndarray]] = {}

    for key, value in history.items():

        if key == "rho":

            result[key] = value

        else:

            result[key] = np.asarray(value)

    result["J"] = np.asarray(J)

    result["Omega_eff_ev"] = np.asarray(Omega_eff_ev)

    result["T_eff_fs"] = np.asarray(T_eff_fs)

    result["tX_fs"] = np.asarray(tX_fs)

    return result





# -----------------------------------------------------------------------------

# VARREDURA NO REGIME DE PERTURBACAO

# -----------------------------------------------------------------------------

def run_sweep_ratio_V0_2J(

    phys: PhysicalParameters,

    num: NumericalParameters,

    ratios: Tuple[float, ...],

    outdir: str,

) -> Dict[str, np.ndarray]:

    """

    Varre o parametro adimensional eta = V0/(2J).

    Para cada eta, executa a mesma dinamica espacial da simulacao principal,

    alterando somente a amplitude V0 da gaussiana:

        V0 = eta * 2J.

    O calculo preserva a geometria original do potencial, inclusive

    x_barrier e width. Isso permite interpretar a varredura como uma

    continuacao controlada do caso publicado, e nao como uma nova simulacao

    com parametros espaciais diferentes.

    Os observaveis principais sao:

      - P3_max: transferencia maxima para a camada |3>

      - P1_min: minima populacao de |1>

      - C_max: maximo de |rho13|

      - purity_min: minima pureza da matriz reduzida

      - t_P3max: instante de maxima transferencia

      - F_X: P3 no instante teorico t_X = T_eff/2

    """

    os.makedirs(outdir, exist_ok=True)

    m_eff = phys.m_eff_factor * phys.m_el

    hbar2_2m_ev_nm2 = (phys.hbar_ev * 1e9) ** 2 / (

        2 * m_eff * 1e-30 / phys.e_charge

    )

    J = phys.Delta ** 2 / phys.U

    Omega_eff_ev = 2.0 * J

    T_eff_fs = (2.0 * np.pi * phys.hbar_ev / Omega_eff_ev) * 1e15

    tX_fs = T_eff_fs / 2.0

    dx = num.WX / num.nx

    dy = num.WY / num.ny

    x = np.linspace(-num.WX / 2.0, num.WX / 2.0, num.nx)

    y = np.linspace(-num.WY / 2.0, num.WY / 2.0, num.ny)

    X, Y = np.meshgrid(x, y, indexing="ij")

    kx = np.fft.fftfreq(num.nx, d=dx) * 2.0 * np.pi

    ky = np.fft.fftfreq(num.ny, d=dy) * 2.0 * np.pi

    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    K2 = KX ** 2 + KY ** 2

    cdt = num.sweep_dt_fs * 1e-15

    dt_hbar = cdt / phys.hbar_ev

    T_half = np.exp(-1j * (hbar2_2m_ev_nm2 * K2) * (dt_hbar / 2.0))

    # O mesmo estado inicial da simulacao principal.

    psi1_initial = np.exp(

        -(X ** 2 + Y ** 2) / (2.0 * num.sigma ** 2)

    ).astype(np.complex128)

    psi1_initial /= np.sqrt(np.sum(np.abs(psi1_initial) ** 2) * dx * dy)

    print("\n" + "=" * 80)

    print("VARREDURA ADIMENSIONAL: eta = V0/(2J)")

    print("=" * 80)

    print(f"J = {J * 1000:.6f} meV")

    print(f"2J = {2.0 * J * 1000:.6f} meV")

    print(f"T_eff = {T_eff_fs:.3f} fs | t_X = {tX_fs:.3f} fs")

    print(f"dt da varredura = {num.sweep_dt_fs:.3f} fs | janela = {num.sweep_nt*num.sweep_dt_fs:.1f} fs")

    print(f"Valores de eta: {ratios}")

    print("=" * 80)

    results = {

        "eta": [],

        "V0_meV": [],

        "P3_max": [],

        "P1_min": [],

        "coherence_max": [],

        "purity_min": [],

        "F_X": [],

        "t_P3max_fs": [],

    }

    # A janela da varredura e deliberadamente menor que a simulacao

    # principal, mas cobre t_X = T_eff/2 e o primeiro maximo da troca.

    sweep_nt = min(num.sweep_nt, num.nt)

    for eta in ratios:

        V0 = float(eta) * 2.0 * J

        if phys.use_potential:

            V = V0 * np.exp(

                -((X - phys.x_barrier) ** 2) / (2.0 * phys.width ** 2)

            )

        else:

            V = np.zeros((num.nx, num.ny))

        # Operador local 2x2 para este V0.

        # Forma vetorizada: evita o duplo loop Python sobre a grade 64x64

        # para cada ponto da varredura.

        v_half = V / 2.0

        Omega = np.sqrt(v_half ** 2 + J ** 2)

        cos_O = np.cos(Omega * dt_hbar)

        sin_O = np.sin(Omega * dt_hbar)

        phase = np.exp(-1j * v_half * dt_hbar)

        U00 = phase * (

            cos_O - 1j * (v_half / Omega) * sin_O

        )

        U01 = phase * (

            -1j * (J / Omega) * sin_O

        )

        U10 = U01.copy()

        U11 = phase * (

            cos_O + 1j * (v_half / Omega) * sin_O

        )

        psi1 = psi1_initial.copy()

        psi3 = np.zeros_like(psi1)

        # Para a varredura, basta guardar os observaveis.

        times = []

        p1s = []

        p3s = []

        coherences = []

        purities = []

        for it in range(sweep_nt + 1):

            if it % num.save_step == 0:

                rho, purity, _ = reduced_density_matrix(

                    psi1, psi3, dx, dy

                )

                times.append(it * num.sweep_dt_fs)

                p1s.append(float(np.real(rho[0, 0])))

                p3s.append(float(np.real(rho[1, 1])))

                coherences.append(float(np.abs(rho[0, 1])))

                purities.append(float(purity))

            if it == sweep_nt:

                break

            psi1 = ifft2_opt(fft2_opt(psi1) * T_half)

            psi3 = ifft2_opt(fft2_opt(psi3) * T_half)

            psi1_new = U00 * psi1 + U01 * psi3

            psi3_new = U10 * psi1 + U11 * psi3

            psi1, psi3 = psi1_new, psi3_new

            psi1 = ifft2_opt(fft2_opt(psi1) * T_half)

            psi3 = ifft2_opt(fft2_opt(psi3) * T_half)

        times = np.asarray(times)

        p1s = np.asarray(p1s)

        p3s = np.asarray(p3s)

        coherences = np.asarray(coherences)

        purities = np.asarray(purities)

        i_max = int(np.argmax(p3s))

        # O ponto mais proximo do t_X teorico.

        i_x = int(np.argmin(np.abs(times - tX_fs)))

        results["eta"].append(float(eta))

        results["V0_meV"].append(V0 * 1000.0)

        results["P3_max"].append(float(np.max(p3s)))

        results["P1_min"].append(float(np.min(p1s)))

        results["coherence_max"].append(float(np.max(coherences)))

        results["purity_min"].append(float(np.min(purities)))

        results["F_X"].append(float(p3s[i_x]))

        results["t_P3max_fs"].append(float(times[i_max]))

        print(

            f"eta={eta:6.2f} | V0={V0*1000:8.4f} meV | "

            f"P3_max={p3s[i_max]:.6f} | P1_min={np.min(p1s):.6f} | "

            f"C_max={np.max(coherences):.6f} | "

            f"purity_min={np.min(purities):.6f} | "

            f"F_X={p3s[i_x]:.6f}"

        )

    for key in results:

        results[key] = np.asarray(results[key], dtype=float)

    # Dados tabulados para analise posterior.

    datafile = os.path.join(outdir, "varredura_V0_sobre_2J.dat")

    np.savetxt(

        datafile,

        np.column_stack([

            results["eta"],

            results["V0_meV"],

            results["P3_max"],

            results["P1_min"],

            results["coherence_max"],

            results["purity_min"],

            results["F_X"],

            results["t_P3max_fs"],

        ]),

        header=(

            "eta_V0_over_2J V0_meV P3_max P1_min "

            "coherence_max purity_min F_X_at_tX t_P3max_fs"

        ),

    )

    # Mostra TODOS os resultados numericos da varredura diretamente no terminal.

    print("\n" + "=" * 110)

    print("RESULTADO COMPLETO DA VARREDURA  eta = V0/(2J)")

    print("=" * 110)

    print(

        f"{'eta':>8} {'V0 (meV)':>12} {'P3_max':>12} {'P1_min':>12} "

        f"{'|rho13|_max':>15} {'purity_min':>14} {'F_X':>12} {'t_P3max (fs)':>16}"

    )

    print("-" * 110)

    for i in range(len(results["eta"])):

        print(

            f"{results['eta'][i]:8.2f} "

            f"{results['V0_meV'][i]:12.5f} "

            f"{results['P3_max'][i]:12.6f} "

            f"{results['P1_min'][i]:12.6f} "

            f"{results['coherence_max'][i]:15.6f} "

            f"{results['purity_min'][i]:14.6f} "

            f"{results['F_X'][i]:12.6f} "

            f"{results['t_P3max_fs'][i]:16.3f}"

        )

    print("=" * 110)

    print("A figura da varredura sera exibida na tela com plt.show().")

    print("=" * 110)

    # Relatorio textual da varredura.

    reportfile = os.path.join(outdir, "relatorio_varredura_V0_sobre_2J.txt")

    with open(reportfile, "w", encoding="utf-8") as f:

        f.write("VARREDURA DO REGIME DE PERTURBACAO\n")

        f.write("=" * 80 + "\n")

        f.write("Parametro de controle: eta = V0/(2J)\n")

        f.write(f"Delta = {phys.Delta*1000:.6f} meV\n")

        f.write(f"U = {phys.U*1000:.6f} meV\n")

        f.write(f"J = {J*1000:.6f} meV\n")

        f.write(f"2J = {2*J*1000:.6f} meV\n")

        f.write(f"x_barrier = {phys.x_barrier:.3f} nm\n")

        f.write(f"width = {phys.width:.3f} nm\n")

        f.write(f"sigma_wavepacket = {num.sigma:.3f} nm\n")

        f.write(f"T_eff = {T_eff_fs:.6f} fs\n")

        f.write(f"t_X = {tX_fs:.6f} fs\n")

        f.write(f"sweep_dt = {num.sweep_dt_fs:.3f} fs | sweep_nt = {sweep_nt} | "

                f"sweep_window = {sweep_nt*num.sweep_dt_fs:.3f} fs\n")

        f.write("\n")

        f.write(

            "eta\tV0(meV)\tP3_max\tP1_min\t|rho13|_max\t"

            "purity_min\tF_X\tt_P3max(fs)\n"

        )

        for i in range(len(results["eta"])):

            f.write(

                f"{results['eta'][i]:.6f}\t"

                f"{results['V0_meV'][i]:.6f}\t"

                f"{results['P3_max'][i]:.8f}\t"

                f"{results['P1_min'][i]:.8f}\t"

                f"{results['coherence_max'][i]:.8f}\t"

                f"{results['purity_min'][i]:.8f}\t"

                f"{results['F_X'][i]:.8f}\t"

                f"{results['t_P3max_fs'][i]:.6f}\n"

            )

    return results



def figure_sweep_V0_2J(

    sweep: Dict[str, np.ndarray],

    phys: PhysicalParameters,

    outdir: str,

    style: FigureStyle,

    show: bool = False,

) -> str:

    """

    Figura principal da varredura V0/(2J).

    Painel (a): maxima transferencia P3 e minima P1.

    Painel (b): maximo de |rho13| e minima pureza.

    O eixo x e adimensional e identifica explicitamente os regimes:

    V0 << 2J, V0 \~ 2J e V0 >> 2J.

    """

    eta = sweep["eta"]

    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.35), dpi=300)

    fig.patch.set_alpha(0)

    ax = axs[0]

    ax.plot(

        eta, sweep["P3_max"],

        color=style.pop_3, lw=style.lw_main,

        marker="o", markersize=4.8,

        label=r"$**\m**ax_t P_3$"

    )

    ax.plot(

        eta, sweep["P1_min"],

        color=style.pop_1, lw=style.lw_main,

        marker="s", markersize=4.5,

        linestyle="--",

        label=r"$**\m**in_t P_1$"

    )

    ax.axvline(1.0, color="black", lw=0.9, linestyle=":", alpha=0.7)

    ax.axvspan(0.0, 1.0, alpha=0.06)

    ax.axvspan(1.0, eta[-1], alpha=0.035)

    ax.text(0.32, 0.92, r"$V_0<2J$", transform=ax.transAxes, fontsize=12)

    ax.text(0.57, 0.92, r"$V_0>2J$", transform=ax.transAxes, fontsize=12)

    # Destaca o caso original, que para V0=3 meV e 2J=0.5 meV corresponde a eta=6.

    original_eta = phys.V0 / (2.0 * (phys.Delta ** 2 / phys.U))

    if np.min(np.abs(eta - original_eta)) < 1e-10:

        idx0 = int(np.argmin(np.abs(eta - original_eta)))

        ax.plot(

            eta[idx0], sweep["P3_max"][idx0],

            marker="o", markersize=8.0,

            markerfacecolor="none", markeredgecolor="black",

            markeredgewidth=1.0,

        )

    panel_label(ax, "(a)")

    ax.set_xlabel(r"$V_0/(2J)$")

    ax.set_ylabel(r"Population")

    ax.set_xlim(eta[0], eta[-1])

    ax.set_ylim(-0.05, 1.05)

    ax.legend(loc="best", fontsize=12)

    polish_axes(ax, grid=True)

    ax = axs[1]

    ax.plot(

        eta, sweep["coherence_max"],

        color=style.coherence, lw=style.lw_main,

        marker="o", markersize=4.8,

        label=r"$**\m**ax_t|**\r**ho_{13}|$"

    )

    ax.plot(

        eta, sweep["purity_min"],

        color=style.purity, lw=style.lw_main,

        marker="s", markersize=4.5,

        linestyle="--",

        label=r"$**\m**in_t**,\m**athrm{Tr}(**\r**ho^2)$"

    )

    ax.axvline(1.0, color="black", lw=0.9, linestyle=":", alpha=0.7)

    panel_label(ax, "(b)")

    ax.set_xlabel(r"$V_0/(2J)$")

    ax.set_ylabel(r"Coherence / purity")

    ax.set_xlim(eta[0], eta[-1])

    ax.set_ylim(-0.05, 1.05)

    ax.legend(loc="best", fontsize=12)

    polish_axes(ax, grid=True)

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.205, top=0.965, wspace=0.27)

    for a in axs:

        prepare_axes_for_export(a)

    filename = os.path.join(outdir, "Fig_varredura_V0_sobre_2J.pdf")

    save_publication_figure(fig, filename, png_dpi=500)

    if show:

        plt.figure(fig.number)

        plt.show(block=True)

    plt.close(fig)

    return filename



# -----------------------------------------------------------------------------

# FIGURAS

# -----------------------------------------------------------------------------

def save_numerical_data(history: Dict[str, np.ndarray | List[np.ndarray]], outdir: str) -> str:

    datafile = os.path.join(outdir, "dados_densidade_reduzida.dat")

    np.savetxt(

        datafile,

        np.column_stack([

            history["t"], history["p1"], history["p3"],

            history["rho13_re"], history["rho13_im"], history["coherence"],

            history["purity"], history["rx"], history["ry"], history["rz"],

        ]),

        header="t_fs P1 P3 Re_rho13 Im_rho13 abs_rho13 purity rx ry rz",

    )

    return datafile



def analytical_curves(

    t_fs: np.ndarray,

    phys: PhysicalParameters,

    history: Dict[str, np.ndarray | List[np.ndarray]],

) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    J = float(history["J"])

    Omega_eff_ev = float(history["Omega_eff_ev"])

    theta = J * t_fs * 1e-15 / phys.hbar_ev

    P1 = np.cos(theta) ** 2

    P3 = np.sin(theta) ** 2

    coherence = 0.5 * np.abs(np.sin(Omega_eff_ev * t_fs * 1e-15 / phys.hbar_ev))

    purity = np.ones_like(t_fs)

    return P1, P3, coherence, purity



def figure_population_dynamics(

    history: Dict[str, np.ndarray | List[np.ndarray]],

    phys: PhysicalParameters,

    outdir: str,

    style: FigureStyle,

    show: bool = False,

) -> str:

    """Figura grande apenas com as populações P1 e P3, em escala de ps."""

    t_ps = history["t"] / 1000.0

    P1_th, P3_th, _, _ = analytical_curves(history["t"], phys, history)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    fig.patch.set_alpha(0)

    ax.set_box_aspect(0.55)

    ax.plot(t_ps, history["p1"], color=style.pop_1, lw=style.lw_main, linestyle="solid")

    ax.plot(t_ps, history["p3"], color=style.pop_3, lw=style.lw_main, linestyle="dashed")

    #ax.plot(t_ps, P1_th, color=style.theory, lw=style.lw_theory, linestyle="--", alpha=0.85)

    #ax.plot(t_ps, P3_th, color=style.theory, lw=style.lw_theory, linestyle=":", alpha=0.85)

#    panel_label(ax, "(a)")

    ax.set_xlabel(r"Time (ps)", labelpad=12)

    ax.set_ylabel(r"Population")

    ax.set_xlim(t_ps[0], t_ps[-1])

    ax.set_ylim(-0.05, 1.0)

    polish_axes(ax, grid=True)

    prepare_axes_for_export(ax)

    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.24, top=0.965)

    filename = os.path.join(outdir, "Fig_populacoes.pdf")

    save_publication_figure(fig, filename, png_dpi=450)

    if show:

        plt.figure(fig.number)

        plt.show(block=True)

    plt.close(fig)

    return filename



def figure_coherence_purity(

    history: Dict[str, np.ndarray | List[np.ndarray]],

    phys: PhysicalParameters,

    outdir: str,

    style: FigureStyle,

    show: bool = False,

) -> str:

    """Figura grande com coerência e pureza no mesmo eixo, em escala de ps."""

    t_ps = history["t"] / 1000.0

    _, _, coh_th, purity_th = analytical_curves(history["t"], phys, history)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    fig.patch.set_alpha(0)

    ax.set_box_aspect(0.55)

    ax.plot(t_ps, history["coherence"], color=style.coherence, lw=style.lw_main, linestyle="solid")

    ax.plot(t_ps, coh_th, color=style.theory, lw=style.lw_theory, linestyle="--", alpha=0.80)

    ax.plot(t_ps, history["purity"], color=style.purity, lw=style.lw_main, linestyle="solid")

    ax.plot(t_ps, purity_th, color=style.purity_theory, lw=style.lw_theory, linestyle=":", alpha=0.90)

#    panel_label(ax, "(b)")

    ax.set_xlabel(r"Time (ps)", labelpad=12)

    ax.set_ylabel(r"Coherence / purity")

    ax.set_xlim(t_ps[0], t_ps[-1])

    ax.set_ylim(-0.05, 1.2)

    polish_axes(ax, grid=True)

    prepare_axes_for_export(ax)

    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.24, top=0.965)

    filename = os.path.join(outdir, "Fig_coerencia_pureza.pdf")

    save_publication_figure(fig, filename, png_dpi=450)

    if show:

        plt.figure(fig.number)

        plt.show(block=True)

    plt.close(fig)

    return filename



def figure_bloch_vector(

    history: Dict[str, np.ndarray | List[np.ndarray]],

    phys: PhysicalParameters,

    outdir: str,

    style: FigureStyle,

    show: bool = False,

) -> str:

    """Figura grande com os componentes do vetor de Bloch, em escala de ps."""

    t_ps = history["t"] / 1000.0

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    fig.patch.set_alpha(0)

    ax.set_box_aspect(0.55)

    ax.plot(t_ps, history["rx"], color=style.bloch_x, lw=style.lw_main, linestyle="solid",label=r"$r_x$")

    ax.plot(t_ps, history["ry"], color=style.bloch_y, lw=style.lw_main, linestyle="dashed",label=r"$r_y$")

    ax.plot(t_ps, history["rz"], color=style.bloch_z, lw=style.lw_main, linestyle="dotted",label=r"$r_z$")

#    panel_label(ax, "(c)")

    ax.set_xlabel(r"Time (ps)", labelpad=12)

    ax.set_ylabel(r"Bloch vector")

    ax.set_xlim(t_ps[0], t_ps[-1])

    ax.set_ylim(-1.05, 2.5)

    polish_axes(ax, grid=True)

    ax.legend(

    loc="upper right",

    frameon=False,

    fontsize=15

)

    prepare_axes_for_export(ax)

    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.24, top=0.965)

    filename = os.path.join(outdir, "Fig_bloch_vector.pdf")

    save_publication_figure(fig, filename, png_dpi=450)

    if show:

        plt.figure(fig.number)

        plt.show(block=True)

    plt.close(fig)

    return filename

def select_representative_states(

    history: Dict[str, np.ndarray | List[np.ndarray]],

) -> Dict[str, Dict[str, np.ndarray | float | int]]:

    t = history["t"]

    T_eff = float(history["T_eff_fs"])

    targets = {

        r"$0$": 0.0,

        r"$T_{**\m**athrm{eff}}/4$": T_eff / 4.0,

        r"$t_X=T_{**\m**athrm{eff}}/2$": T_eff / 2.0,

        r"$3T_{**\m**athrm{eff}}/4$": 3.0 * T_eff / 4.0,

        r"$T_{**\m**athrm{eff}}$": T_eff,

    }

    selected = {}

    for label, target in targets.items():

        idx = int(np.argmin(np.abs(t - target)))

        rho = history["rho"][idx]

        bloch = np.array([history["rx"][idx], history["ry"][idx], history["rz"][idx]], dtype=float)

        selected[label] = {"idx": idx, "t": float(t[idx]), "rho": rho, "bloch": bloch}

    return selected



def plot_density_bars(ax, matrix: np.ndarray, component: str, zlim: Tuple[float, float], tick_fontsize: float = 7.0) -> None:

    """Plota barras 3D para Re(rho) ou Im(rho) em formato limpo de publicação."""

    data = np.real(matrix) if component == "real" else np.imag(matrix)

    xpos, ypos = np.meshgrid(np.arange(2), np.arange(2), indexing="ij")

    xpos = xpos.ravel()

    ypos = ypos.ravel()

    zpos = np.zeros_like(xpos, dtype=float)

    dx = dy = 0.64 * np.ones_like(xpos, dtype=float)

    dz = data.ravel()

    norm = Normalize(vmin=zlim[0], vmax=zlim[1])

    colors = cm.coolwarm(norm(dz))

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, edgecolor="black", linewidth=0.35, shade=True)

    ax.set_xticks([0.32, 1.32])

    ax.set_yticks([0.32, 1.32])

    ax.set_xticklabels([r"$|1**\r**angle$", r"$|3**\r**angle$"], fontsize=tick_fontsize)

    ax.set_yticklabels([r"$**\l**angle1|$", r"$**\l**angle3|$"], fontsize=tick_fontsize)

    ax.set_zticks([-1, 0, 1])

    ax.set_zticklabels([r"$-1$", r"$0$", r"$1$"], fontsize=tick_fontsize)

    ax.set_zlim(-1.0, 1.0)

    ax.view_init(elev=27, azim=-48)

    ax.set_box_aspect((1, 1, 0.74))

    ax.tick_params(pad=0, labelsize=tick_fontsize)

    ax.grid(False)

def figure_density_tomography(

    history: Dict[str, np.ndarray | List[np.ndarray]],

    outdir: str,

    show: bool = False,

) -> str:

    selected = select_representative_states(history)

    fig = plt.figure(figsize=(9.2, 4.90))

    fig.subplots_adjust(left=0.045, right=0.992, bottom=0.045, top=0.87, wspace=0.00, hspace=0.10)

    labels = list(selected.keys())

    display_labels = [r"$0$", r"$T_{**\m**athrm{eff}}/4$", r"$t_X$", r"$T_{**\m**athrm{eff}}$"]

    for col, label in enumerate(labels):

        item = selected[label]

        rho = item["rho"]

        t_sel = item["t"]

        ax_re = fig.add_subplot(2, 4, col + 1, projection="3d")

        plot_density_bars(ax_re, rho, "real", zlim=(-1.0, 1.0), tick_fontsize=5.4)

        ax_re.set_title(f"{display_labels[col]}\n{t_sel:.0f} fs", fontsize=8.0, pad=0)

        if col == 0:

            ax_re.text2D(-0.04, 0.93, "(a)", transform=ax_re.transAxes, fontsize=9.0, fontweight="normal")

            ax_re.text2D(-0.11, 0.50, r"$**\m**athrm{Re}(**\r**ho)$", transform=ax_re.transAxes, fontsize=8.0, rotation=90, va="center")

        ax_im = fig.add_subplot(2, 4, col + 5, projection="3d")

        plot_density_bars(ax_im, rho, "imag", zlim=(-0.55, 0.55), tick_fontsize=5.4)

        if col == 0:

            ax_im.text2D(-0.04, 0.93, "(b)", transform=ax_im.transAxes, fontsize=9.0, fontweight="normal")

            ax_im.text2D(-0.11, 0.50, r"$**\m**athrm{Im}(**\r**ho)$", transform=ax_im.transAxes, fontsize=8.0, rotation=90, va="center")

    filename = os.path.join(outdir, "Fig_tomografia_matriz_densidade.pdf")

    fig.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.02)

    fig.savefig(filename.replace(".pdf", ".png"), format="png", dpi=450, bbox_inches="tight", pad_inches=0.02)

    if show:

        plt.figure(fig.number)

        plt.show(block=True)

    plt.close(fig)

    return filename



def figure_bloch_spheres(

    history: Dict[str, np.ndarray | List[np.ndarray]],

    outdir: str,

    show: bool = False,

) -> str | None:

    if not QISKIT_AVAILABLE:

        return None

    selected = select_representative_states(history)

    temp_files = []

    # O Qiskit gera uma figura por esfera. Salvamos e depois montamos uma prancha.

    for k, (label, item) in enumerate(selected.items()):

        bloch = item["bloch"]

        t_sel = item["t"]

        fig_bloch = plot_bloch_vector(bloch, title=f"{label}, t={t_sel:.0f} fs")

        temp = os.path.join(outdir, f"_tmp_bloch_{k}.png")

        fig_bloch.savefig(temp, dpi=300, bbox_inches="tight", pad_inches=0.02)

        plt.close(fig_bloch)

        temp_files.append((temp, label, t_sel))

    fig, axs = plt.subplots(1, 4, figsize=(8.8, 2.65))

    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.03, top=0.86, wspace=0.02)

    for i, (temp, label, t_sel) in enumerate(temp_files):

        img = plt.imread(temp)

        axs[i].imshow(img)

        axs[i].axis("off")

        axs[i].set_title(f"{label}\n$t={t_sel:.0f}$ fs", fontsize=8.0)

        axs[i].text(0.02, 0.98, f"({chr(97 + i)})", transform=axs[i].transAxes,

                    ha="left", va="top", fontsize=9.0, fontweight="normal")

    filename = os.path.join(outdir, "Fig_tomografia_esferas_bloch.pdf")

    fig.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.02)

    fig.savefig(filename.replace(".pdf", ".png"), format="png", dpi=450, bbox_inches="tight", pad_inches=0.02)

    if show:

        plt.figure(fig.number)

        plt.show(block=True)

    plt.close(fig)

    for temp, _, _ in temp_files:

        try:

            os.remove(temp)

        except OSError:

            pass

    return filename



def plot_bloch_sphere_custom(

    ax,

    bloch: np.ndarray,

    vector_color: str = "#b2182b",

    sphere_color: str = "#7f7f7f",

) -> None:

    """Plota uma esfera de Bloch simples diretamente no eixo 3D fornecido.

    Esta versao evita depender do Qiskit para montar a prancha final e,

    principalmente, permite remover titulos internos; todas as descricoes

    ficam no caption gerado no terminal/relatorio.

    """

    u = np.linspace(0, 2 * np.pi, 42)

    v = np.linspace(0, np.pi, 22)

    xs = np.outer(np.cos(u), np.sin(v))

    ys = np.outer(np.sin(u), np.sin(v))

    zs = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_wireframe(xs, ys, zs, rstride=4, cstride=4, color=sphere_color, linewidth=0.25, alpha=0.35)

    ax.plot([-1, 1], [0, 0], [0, 0], color=sphere_color, lw=0.35, alpha=0.55)

    ax.plot([0, 0], [-1, 1], [0, 0], color=sphere_color, lw=0.35, alpha=0.55)

    ax.plot([0, 0], [0, 0], [-1, 1], color=sphere_color, lw=0.35, alpha=0.55)

    ax.quiver(

        0, 0, 0,

        float(bloch[0]), float(bloch[1]), float(bloch[2]),

        length=1.0,

        normalize=False,

        color=vector_color,

        linewidth=1.2,

        arrow_length_ratio=0.16,

    )

    ax.text(1.13, 0, 0, r"$x$", fontsize=10, ha="center", va="center")

    ax.text(0, 1.13, 0, r"$y$", fontsize=10, ha="center", va="center")

    ax.text(0, 0, 1.13, r"$z$", fontsize=10, ha="center", va="center")

    ax.set_xlim(-1.05, 1.05)

    ax.set_ylim(-1.05, 1.05)

    ax.set_zlim(-1.05, 1.05)

    ax.set_box_aspect((1, 1, 1))

    ax.view_init(elev=22, azim=-38)

    ax.set_axis_off()



def figure_tomography_bloch_pairs(

    history: Dict[str, np.ndarray | List[np.ndarray]],

    outdir: str,

    style: FigureStyle,

    show: bool = False,

) -> str:

    """Figura combinada: para cada tempo, Re(rho), Im(rho) e esfera de Bloch.

    Estrutura dos subplots:

    (a) t = 0                  Re(rho) | Im(rho) | Bloch

    (b) t = T_eff/4            Re(rho) | Im(rho) | Bloch

    (c) t = t_X = T_eff/2      Re(rho) | Im(rho) | Bloch

    (d) t = 3T_eff/4           Re(rho) | Im(rho) | Bloch

    (e) t = T_eff              Re(rho) | Im(rho) | Bloch

    Nao ha titulos sobre os paineis; a identificacao completa fica no caption.

    """

    selected = select_representative_states(history)

    labels = list(selected.keys())

    fig = plt.figure(figsize=(7.2, 8.8))

    gs = fig.add_gridspec(

        nrows=len(labels),

        ncols=3,

        left=0.035,

        right=0.985,

        bottom=0.025,

        top=0.985,

        wspace=0.00,

        hspace=0.03,

        width_ratios=[1.0, 1.0, 0.95],

    )

    panel_letters = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    for row, label in enumerate(labels):

        item = selected[label]

        rho = item["rho"]

        bloch = item["bloch"]

        ax_re = fig.add_subplot(gs[row, 0], projection="3d")

        plot_density_bars(ax_re, rho, "real", zlim=(-1.0, 1.0), tick_fontsize=5.0)

        ax_re.text2D(

            -0.02, 0.95, panel_letters[row],

            transform=ax_re.transAxes,

            fontsize=9.2,

            fontweight="normal",

            ha="left",

            va="top",

        )

        if row == 0:

            ax_re.text2D(0.36, 1.02, r"$**\m**athrm{Re}(**\r**ho)$", transform=ax_re.transAxes,

                         fontsize=7.2, ha="center", va="bottom")

        ax_im = fig.add_subplot(gs[row, 1], projection="3d")

        plot_density_bars(ax_im, rho, "imag", zlim=(-0.55, 0.55), tick_fontsize=5.0)

        if row == 0:

            ax_im.text2D(0.36, 1.02, r"$**\m**athrm{Im}(**\r**ho)$", transform=ax_im.transAxes,

                         fontsize=7.2, ha="center", va="bottom")

        ax_bl = fig.add_subplot(gs[row, 2], projection="3d")

        plot_bloch_sphere_custom(ax_bl, bloch, vector_color=style.bloch_y)

        if row == 0:

            ax_bl.text2D(0.50, 1.02, r"Bloch", transform=ax_bl.transAxes,

                         fontsize=7.2, ha="center", va="bottom")

    filename = os.path.join(outdir, "Fig_tomografia_bloch_pares.pdf")

    fig.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.02)

    fig.savefig(filename.replace(".pdf", ".png"), format="png", dpi=450, bbox_inches="tight", pad_inches=0.02)

    if show:

        plt.figure(fig.number)

        plt.show(block=True)

    plt.close(fig)

    return filename





def _safe_filename_label(label: str) -> str:

    """Converte rótulos LaTeX dos tempos em nomes curtos para arquivos."""

    if "3T" in label:

        return "d_3T4"

    if "t_X" in label:

        return "c_tX"

    if "/4" in label:

        return "b_T4"

    if label.strip() == r"$0$":

        return "a_t0"

    return "e_Teff"



def _display_time_label(label: str) -> str:

    """Rótulos compactos dos tempos para inserir discretamente na figura."""

    if "3T" in label:

        return r"$t=3T_{**\m**athrm{eff}}/4$"

    if "t_X" in label:

        return r"$t=t_X=T_{**\m**athrm{eff}}/2$"

    if "/4" in label:

        return r"$t=T_{**\m**athrm{eff}}/4$"

    if label.strip() == r"$0$":

        return r"$t=0$"

    return r"$t=T_{**\m**athrm{eff}}$"



def _panel_letter_from_index(index: int) -> str:

    return f"({chr(97 + index)})"







def _crop_png_white_border(path: str, pad: int = 4) -> None:

    """Remove margens brancas de um PNG, preservando pequena borda."""

    try:

        img = Image.open(path).convert("RGBA")

        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))

        diff = ImageChops.difference(img, bg)

        bbox = diff.getbbox()

        if bbox is None:

            return

        left, upper, right, lower = bbox

        left = max(left - pad, 0)

        upper = max(upper - pad, 0)

        right = min(right + pad, img.size[0])

        lower = min(lower + pad, img.size[1])

        img.crop((left, upper, right, lower)).save(path)

    except Exception:

        pass

def _qiskit_bloch_to_png(bloch: np.ndarray, outdir: str, key: str) -> str | None:

    """

    Gera a esfera padrão do Qiskit como PNG temporário.

    Ajustes editoriais:

    - sem título interno acima da esfera;

    - fontes maiores para os rótulos da esfera, especialmente |0>, |1>, x e y;

    - bbox bem justo para reduzir espaços vazios ao inserir no painel final.

    """

    if not QISKIT_AVAILABLE:

        return None

    fig_bloch = plot_bloch_vector([float(bloch[0]), float(bloch[1]), float(bloch[2])], title="")

    # O Qiskit cria textos nos eixos 3D. Aqui aumentamos todos os rótulos

    # visíveis da esfera, incluindo |0>, |1>, x e y.

    for ax in fig_bloch.axes:

        try:

            ax.set_title("")

        except Exception:

            pass

        for txt in getattr(ax, "texts", []):

            txt.set_fontsize(18)

            txt.set_fontweight("regular")

        try:

            ax.tick_params(labelsize=13)

        except Exception:

            pass

    temp = os.path.join(outdir, f"_tmp_qiskit_bloch_{key}.png")

    fig_bloch.savefig(temp, dpi=440, bbox_inches="tight", pad_inches=0.0)

    plt.close(fig_bloch)

    _crop_png_white_border(temp, pad=2)

    return temp



def figure_tomography_bloch_pair_files(

    history: Dict[str, np.ndarray | List[np.ndarray]],

    outdir: str,

    style: FigureStyle,

    show: bool = False,

) -> Dict[str, str]:

    """Gera cinco PDFs independentes, cada um com Re(rho), Im(rho) e esfera padrão do Qiskit.

    Ajustes editoriais:

    - sem rótulo de tempo dentro da figura;

    - sem títulos internos sobre Re(rho), Im(rho) ou Bloch;

    - letra (a)--(e) posicionada acima da tomografia real;

    - margens reduzidas e melhor uso horizontal do espaço;

    - esfera padrão do Qiskit mantida, com fontes ampliadas.

    """

    selected = select_representative_states(history)

    output: Dict[str, str] = {}

    # Figura larga, baixa e compacta: adequada para coluna larga ou figure*,

    # mas cada PDF fica legível individualmente.

    fig_width = 7.20

    fig_height = 1.85

    for idx, (label, item) in enumerate(selected.items()):

        rho = item["rho"]

        bloch = item["bloch"]

        key = _safe_filename_label(label)

        panel = _panel_letter_from_index(idx)

        fig = plt.figure(figsize=(fig_width, fig_height))

        gs = fig.add_gridspec(

            nrows=1,

            ncols=3,

            left=0.015,

            right=0.995,

            bottom=0.030,

            top=0.985,

            width_ratios=[1.00, 1.00, 1.18],

            wspace=0.065,

        )

        ax_re = fig.add_subplot(gs[0, 0], projection="3d")

        plot_density_bars(ax_re, rho, "real", zlim=(-1.0, 1.0), tick_fontsize=7.0)

        ax_re.text2D(

            0.00, 1.015, panel,

            transform=ax_re.transAxes,

            fontsize=18.0,

            fontweight="normal",

            ha="left",

            va="bottom",

        )

        ax_im = fig.add_subplot(gs[0, 1], projection="3d")

        plot_density_bars(ax_im, rho, "imag", zlim=(-1.0, 1.0), tick_fontsize=7.0)

        ax_bl = fig.add_subplot(gs[0, 2])

        ax_bl.axis("off")

        temp_bloch = _qiskit_bloch_to_png(bloch, outdir, key)

        if temp_bloch is not None and os.path.exists(temp_bloch):

            img = plt.imread(temp_bloch)

            ax_bl.imshow(img)

            ax_bl.set_anchor("C")

            try:

                os.remove(temp_bloch)

            except OSError:

                pass

        else:

            ax_bl.remove()

            ax_bl = fig.add_subplot(gs[0, 2], projection="3d")

            plot_bloch_sphere_custom(ax_bl, bloch, vector_color=style.bloch_y)

        filename = os.path.join(outdir, f"Fig_tomografia_bloch_{key}.pdf")

        fig.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.015)

        fig.savefig(filename.replace(".pdf", ".png"), format="png", dpi=520, bbox_inches="tight", pad_inches=0.015)

        if show:

            plt.figure(fig.number)

            plt.show(block=True)

        plt.close(fig)

        output[f"tomografia_bloch_{key}"] = filename

    return output

def compute_gate_report(

    history: Dict[str, np.ndarray | List[np.ndarray]],

) -> Dict[str, float | np.ndarray]:

    selected = select_representative_states(history)

    rho_X = selected[r"$t_X=T_{**\m**athrm{eff}}/2$"]["rho"]

    ket3 = np.array([0.0, 1.0], dtype=complex)

    F_X_from_1_to_3 = float(np.real(np.conjugate(ket3) @ rho_X @ ket3))

    final_idx = -1

    return {

        "F_X_from_1_to_3": F_X_from_1_to_3,

        "purity_final": float(history["purity"][final_idx]),

        "coherence_final": float(history["coherence"][final_idx]),

        "P1_final": float(history["p1"][final_idx]),

        "P3_final": float(history["p3"][final_idx]),

        "rx_final": float(history["rx"][final_idx]),

        "ry_final": float(history["ry"][final_idx]),

        "rz_final": float(history["rz"][final_idx]),

    }



def write_report_and_captions(

    history: Dict[str, np.ndarray | List[np.ndarray]],

    phys: PhysicalParameters,

    num: NumericalParameters,

    figures: Dict[str, str | None],

    datafile: str,

    outdir: str,

) -> str:

    J = float(history["J"])

    T_eff = float(history["T_eff_fs"])

    tX = float(history["tX_fs"])

    gate = compute_gate_report(history)

    caption_dynamics = rf"""\caption{{Coherent dynamics of the effective layer qubit. (a) Populations $P_1$ and $P_3$ of the outer layers. The blue and red curves denote the numerical populations, while the black dashed and dotted curves denote the corresponding analytical predictions. The time axis is expressed in picoseconds.}}"""

    caption_coherence_purity = rf"""\caption{{Coherence and purity of the reduced effective layer qubit. (b) Reduced interlayer coherence $|\rho_{{13}}|$ and purity $\mathrm{{Tr}}(\rho^2)$ shown on the same scale. The green and black curves denote the numerical coherence and purity, respectively; the black dashed and gray dotted curves denote the corresponding analytical results. In the unperturbed regime the purity remains close to unity, while the coherence oscillates between zero and its maximum value $1/2$.}}"""

    caption_bloch_vector = rf"""\caption{{Bloch-vector characterization of the effective layer qubit. (c) Components $r_x=2\mathrm{{Re}}(\rho_{{13}})$, $r_y=-2\mathrm{{Im}}(\rho_{{13}})$, and $r_z=\rho_{{11}}-\rho_{{33}}$, shown as blue, red dashed, and gray dotted curves, respectively. The dynamics is consistent with a coherent rotation generated by the effective Pauli-$X$ Hamiltonian.}}"""

    caption_tomography = rf"""\caption{{State tomography and Bloch-sphere representation of the effective layer qubit. Panels (a)--(e) correspond respectively to $t=0$, $T_{{\mathrm{{eff}}}}/4$, $t_X=T_{{\mathrm{{eff}}}}/2$, $3T_{{\mathrm{{eff}}}}/4$, and $T_{{\mathrm{{eff}}}}$. In each independent panel, the left and middle plots show the real and imaginary parts of the reduced density matrix in perspective in the logical basis ${{|1\rangle,|3\rangle}}$, while the right plot shows the corresponding Qiskit Bloch-sphere representation. The vertical axis of each tomographic plot is scaled with ticks at $-1$, $0$, and $1$.}}"""

    caption_bloch = rf"""\caption{{Bloch-sphere representation of the effective layer qubit at the same times used in the density-matrix tomography. The trajectory corresponds to the coherent evolution induced by the effective interlayer coupling $J=\Delta^2/U$.}}"""

    report_file = os.path.join(outdir, "relatorio_publicacao_e_captions.txt")

    with open(report_file, "w", encoding="utf-8") as f:

        f.write("RELATORIO DA SIMULACAO - VERSAO PUBLICACAO\n")

        f.write("=" * 80 + "\n")

        f.write(f"Qiskit disponivel: {QISKIT_AVAILABLE}\n")

        if not QISKIT_AVAILABLE:

            f.write(f"Erro/aviso Qiskit: {QISKIT_IMPORT_ERROR}\n")

        f.write(f"Delta = {phys.Delta*1000:.6f} meV\n")

        f.write(f"U = {phys.U*1000:.6f} meV\n")

        f.write(f"J = {J*1000:.8f} meV\n")

        f.write(f"V0 = {phys.V0*1000:.6f} meV\n")

        f.write(f"T_eff = {T_eff:.8f} fs\n")

        f.write(f"t_X = {tX:.8f} fs\n")

        f.write(f"Grade = {num.nx} x {num.ny}\n")

        f.write(f"dt = {num.dt_fs:.6f} fs | nt = {num.nt}\n")

        f.write("\nRESULTADOS\n")

        for key, value in gate.items():

            f.write(f"{key} = {value}\n")

        f.write("\nARQUIVOS GERADOS\n")

        f.write(f"dados = {datafile}\n")

        for key, value in figures.items():

            f.write(f"{key} = {value}\n")

        f.write("\nCAPTION - DINAMICA\n")

        f.write(caption_dynamics + "\n\n")

        f.write("CAPTION - COERENCIA E PUREZA\n")

        f.write(caption_coherence_purity + "\n\n")

        f.write("CAPTION - VETOR DE BLOCH\n")

        f.write(caption_bloch_vector + "\n\n")

        f.write("CAPTION - TOMOGRAFIA + ESFERAS DE BLOCH (5 PDFs)\n")

        f.write(caption_tomography + "\n")

    print("\n" + "=" * 80)

    print("RESULTADOS FINAIS")

    print("=" * 80)

    print(f"Fidelidade estimada |1> -> |3> em t_X: F_X = {gate['F_X_from_1_to_3']:.8f}")

    print(f"Pureza final: {gate['purity_final']:.8f}")

    print(f"Coerencia final |rho13|: {gate['coherence_final']:.8f}")

    print(f"Relatorio salvo: {os.path.abspath(report_file)}")

    print("=" * 80)

    print("\n" + "=" * 80)

    print("CAPTION - POPULACOES")

    print("=" * 80)

    print(caption_dynamics)

    print("=" * 80)

    print("\n" + "=" * 80)

    print("CAPTION - COERENCIA E PUREZA")

    print("=" * 80)

    print(caption_coherence_purity)

    print("=" * 80)

    print("\n" + "=" * 80)

    print("CAPTION - VETOR DE BLOCH")

    print("=" * 80)

    print(caption_bloch_vector)

    print("=" * 80)

    print("\n" + "=" * 80)

    print("CAPTION - TOMOGRAFIA + ESFERAS DE BLOCH (5 PDFs)")

    print("=" * 80)

    print(caption_tomography)

    print("=" * 80)

    return report_file



# -----------------------------------------------------------------------------

# EXECUCAO PRINCIPAL

# -----------------------------------------------------------------------------

def main() -> None:
    # Varredura definitiva em eta, usando os parametros numericamente convergidos.
    configure_matplotlib()

    phys = PhysicalParameters()

    # Configuracao convergida:
    # W = 1200 nm, N = 384, dx = 3.125 nm, dt = 0.5 fs.
    num = NumericalParameters(
        nx=384,
        ny=384,
        WX=1200.0,
        WY=1200.0,
        dt_fs=0.5,
        nt=15000,
        save_step=50,
        sigma=15.0,
        show_figures=False,
        sweep_ratios=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0),
        sweep_dt_fs=0.5,
        sweep_nt=9000,
    )

    print("\n" + "=" * 90)
    print("VARREDURA DEFINITIVA EM eta = V0/(2J)")
    print("=" * 90)
    print("Configuracao numericamente convergida:")
    print("W = 1200 nm | N = 384 x 384 | dx = 3.125 nm | dt = 0.5 fs")
    print(f"Delta = {phys.Delta*1000:.6f} meV")
    print(f"U = {phys.U*1000:.6f} meV")
    print(f"J = Delta^2/U = {(phys.Delta**2/phys.U)*1000:.9f} meV")
    print("Varredura: eta = 0, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 12")
    print("V0 = 2 J eta")
    print("=" * 90)

    sweep = run_sweep_ratio_V0_2J(
        phys,
        num,
        num.sweep_ratios,
        outdir="/tmp/tricamada_sweep_eta",
    )

    # Plot only the sweep result; no publication figures and no TeX.
    try:
        style = FigureStyle()
        fig = figure_sweep_V0_2J(
            sweep,
            phys,
            outdir="/tmp/tricamada_sweep_eta",
            style=style,
            show=True,
        )
        if fig is not None:
            plt.show()
    except Exception as exc:
        print(f"\nAviso: nao foi possivel gerar o grafico da varredura: {exc}")
        print("Os resultados numericos da varredura permanecem validos.")

    print("\nVARREDURA CONCLUIDA.")
    print("Nenhuma figura de publicacao foi gerada antes da varredura.")
    print("Nenhuma figura foi salva automaticamente.")


if __name__ == "__main__":

    main()
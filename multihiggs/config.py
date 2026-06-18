from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .term_maps import TermMap, parse_term_maps

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 path
    import tomli as tomllib


DEFAULT_MG_SETTINGS = [
    "set group_subprocesses Auto",
    "set ignore_six_quark_processes False",
    "set low_mem_multicore_nlo_generation False",
    "set complex_mass_scheme False",
    "set include_lepton_initiated_processes False",
    "set gauge unitary",
    "set loop_optimized_output True",
    "set loop_color_flows False",
    "set max_npoint_for_channel 0",
    "set default_unset_couplings 99",
    "set max_t_for_channel 99",
    "set zerowidth_tchannel True",
    "set nlo_mixed_expansion True",
]


DEFAULT_PRE_MODEL_COMMANDS = [
    "import model sm",
    "define p = g u c d s u~ c~ d~ s~",
    "define j = g u c d s u~ c~ d~ s~",
    "define l+ = e+ mu+",
    "define l- = e- mu-",
    "define vl = ve vm vt",
    "define vl~ = ve~ vm~ vt~",
]


DEFAULT_NO_CUT_COMMANDS = [
    "set dsqrt_shat 0.0",
    "set ptheavy 0.0",
    "set pt_min_pdg {}",
    "set pt_max_pdg {}",
    "set eta_min_pdg {}",
    "set eta_max_pdg {}",
    "set mxx_min_pdg {}",
]


DEFAULT_NEVENTS = 10000


@dataclass(frozen=True)
class CouplingConfig:
    name: str
    parameter: str
    fit_name: str
    fit_range: tuple[float, float]
    points: int
    fit_offset: float = 0.0
    sm_value: float = 0.0
    significant_digits: int = 6
    value_format: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CouplingConfig":
        name = str(data["name"])
        fit_range = data.get("range", data.get("fit_range"))
        if fit_range is None or len(fit_range) != 2:
            raise ValueError(f"Coupling {name!r} must define range = [min, max]")
        return cls(
            name=name,
            parameter=str(data.get("parameter", name)),
            fit_name=str(data.get("fit_name", name)),
            fit_range=(float(fit_range[0]), float(fit_range[1])),
            points=int(data.get("points", 1)),
            fit_offset=float(data.get("fit_offset", 0.0)),
            sm_value=float(data.get("sm_value", 0.0)),
            significant_digits=int(data.get("significant_digits", 6)),
            value_format=data.get("value_format"),
        )

    def scan_from_fit(self, value: float) -> float:
        return float(value) - self.fit_offset

    def fit_from_scan(self, value: float) -> float:
        return float(value) + self.fit_offset

    def format_value(self, value: float) -> str:
        if self.value_format:
            return format(float(value), self.value_format)
        rounded = round_sig(float(value), self.significant_digits)
        if abs(rounded) < 1e-12:
            rounded = 0.0
        return str(float(rounded))


@dataclass(frozen=True)
class MGOptions:
    accuracy: float | None = None
    points: int | None = None
    iterations: int | None = None
    extra: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MGOptions":
        data = data or {}
        return cls(
            accuracy=None if data.get("accuracy") is None else float(data["accuracy"]),
            points=None if data.get("points") is None else int(data["points"]),
            iterations=None if data.get("iterations") is None else int(data["iterations"]),
            extra=[str(item) for item in data.get("extra", [])],
        )

    def launch_suffix(self) -> str:
        pieces: list[str] = []
        if self.accuracy is not None:
            pieces.append(f"--accuracy={self.accuracy}")
        if self.points is not None:
            pieces.append(f"--points={self.points}")
        if self.iterations is not None:
            pieces.append(f"--iterations={self.iterations}")
        pieces.extend(self.extra)
        return " ".join(pieces)


@dataclass(frozen=True)
class ScanConfig:
    run_number: str = "1"
    energy_tev: float = 14.0
    nevents: int = DEFAULT_NEVENTS
    min_events: int | None = None
    strategy: str = "chebyshev_lobatto"
    sort: str = "sm_first"
    skip_existing: bool = True
    no_cuts: bool = True
    no_cut_commands: list[str] = field(default_factory=lambda: list(DEFAULT_NO_CUT_COMMANDS))
    extra_set_commands: list[str] = field(default_factory=list)
    madgraph: MGOptions = field(default_factory=MGOptions)

    def __post_init__(self) -> None:
        if self.nevents < 1:
            raise ValueError("[scan] nevents must be at least 1")
        if self.min_events is not None and self.min_events < 0:
            raise ValueError("[scan] min_events must be non-negative")

    @property
    def event_minimum(self) -> int:
        if self.min_events is not None:
            return self.min_events
        return self.nevents

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScanConfig":
        min_events = data.get("min_events")
        return cls(
            run_number=str(data.get("run_number", "1")),
            energy_tev=float(data.get("energy_tev", 14.0)),
            nevents=int(data.get("nevents", DEFAULT_NEVENTS)),
            min_events=None if min_events is None else int(min_events),
            strategy=str(data.get("strategy", "chebyshev_lobatto")),
            sort=str(data.get("sort", "sm_first")),
            skip_existing=bool(data.get("skip_existing", True)),
            no_cuts=bool(data.get("no_cuts", True)),
            no_cut_commands=[str(item) for item in data.get("no_cut_commands", DEFAULT_NO_CUT_COMMANDS)],
            extra_set_commands=[str(item) for item in data.get("extra_set_commands", [])],
            madgraph=MGOptions.from_dict(data.get("madgraph")),
        )


@dataclass(frozen=True)
class FitConfig:
    basis: str
    terms: tuple[tuple[int, ...], ...]
    normalize_to_sm: bool = True
    term_map: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], ndims: int) -> "FitConfig":
        terms = data.get("terms")
        if not terms:
            raise ValueError("The [fit] section must define terms = [[...], ...]")
        clean_terms: list[tuple[int, ...]] = []
        for term in terms:
            if len(term) != ndims:
                raise ValueError(f"Fit term {term!r} does not have {ndims} entries")
            clean_terms.append(tuple(int(power) for power in term))
        return cls(
            basis=str(data.get("basis", "chebyshev")),
            terms=tuple(clean_terms),
            normalize_to_sm=bool(data.get("normalize_to_sm", True)),
            term_map=None if data.get("term_map") is None else str(data["term_map"]),
        )


@dataclass(frozen=True)
class ObservableConfig:
    name: str
    kind: str
    bins: tuple[float, ...]
    pdg_id: int | None = None
    pdg_ids: tuple[int, ...] = ()
    which: str = "all"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservableConfig":
        bins = tuple(float(edge) for edge in data.get("bins", []))
        if len(bins) < 2:
            raise ValueError(f"Observable {data.get('name')!r} must define at least two bin edges")
        return cls(
            name=str(data["name"]),
            kind=str(data["kind"]),
            bins=bins,
            pdg_id=None if data.get("pdg_id") is None else int(data["pdg_id"]),
            pdg_ids=tuple(int(pid) for pid in data.get("pdg_ids", [])),
            which=str(data.get("which", "all")),
        )


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    name: str
    mg5_path: Path
    model: str
    generate: str
    output: str
    settings: tuple[str, ...]
    pre_model_commands: tuple[str, ...]
    post_model_commands: tuple[str, ...]
    couplings: tuple[CouplingConfig, ...]
    scan: ScanConfig
    fit: FitConfig
    extra_points: tuple[dict[str, float], ...] = ()
    observables: tuple[ObservableConfig, ...] = ()
    term_maps: dict[str, TermMap] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: Path) -> "ProjectConfig":
        process = data.get("process", {})
        if not process:
            raise ValueError("Config must contain a [process] section")
        couplings = tuple(CouplingConfig.from_dict(item) for item in data.get("couplings", []))
        if not couplings:
            raise ValueError("Config must contain at least one [[couplings]] entry")
        extra_points = tuple(
            {str(key): float(value) for key, value in point.items()}
            for point in data.get("extra_points", [])
        )
        observables = tuple(
            ObservableConfig.from_dict(item) for item in data.get("observables", [])
        )
        return cls(
            path=path,
            name=str(process.get("name", process.get("output", "process"))),
            mg5_path=Path(process["mg5_path"]).expanduser(),
            model=str(process["model"]),
            generate=str(process["generate"]),
            output=str(process["output"]),
            settings=tuple(str(item) for item in process.get("settings", DEFAULT_MG_SETTINGS)),
            pre_model_commands=tuple(
                str(item) for item in process.get("pre_model_commands", DEFAULT_PRE_MODEL_COMMANDS)
            ),
            post_model_commands=tuple(str(item) for item in process.get("post_model_commands", [])),
            couplings=couplings,
            scan=ScanConfig.from_dict(data.get("scan", {})),
            fit=FitConfig.from_dict(data.get("fit", {}), len(couplings)),
            extra_points=extra_points,
            observables=observables,
            term_maps=parse_term_maps(data.get("term_maps")),
        )

    @property
    def process_dir(self) -> Path:
        output = Path(self.output)
        if output.is_absolute():
            return output
        return self.mg5_path / output

    @property
    def output_dir(self) -> Path:
        return Path("outputs") / self.name

    def coupling_by_name(self, name: str) -> CouplingConfig:
        for coupling in self.couplings:
            if coupling.name == name or coupling.parameter == name:
                return coupling
        raise KeyError(name)


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("rb") as stream:
        data = tomllib.load(stream)
    return ProjectConfig.from_dict(data, config_path)


def round_sig(value: float, sig: int = 6) -> float:
    import math

    if value == 0.0:
        return 0.0
    if math.isnan(value):
        return 0.0
    return round(value, sig - int(math.floor(math.log10(abs(value)))) - 1)

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
FORBIDDEN_ENDINGS = (
    ".pth", ".pt", ".ckpt", ".safetensors", ".onnx", ".bin",
    ".pkl", ".pickle", ".joblib", ".h5", ".hdf5", ".engine", ".plan",
    ".npy", ".npz", ".tif", ".tiff", ".png", ".jpg", ".jpeg",
    ".tar", ".tar.gz", ".tgz", ".zip", ".7z", ".rar", ".gz", ".bz2", ".xz",
    ".pyc", ".pyo",
)


def public_files():
    return [
        path for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.resolve() != THIS_FILE
    ]


def test_no_vendored_source_weights_data_archives_or_private_paths():
    files = public_files()
    assert not any(path.name.lower().endswith(FORBIDDEN_ENDINGS) for path in files)
    assert not any(path.name == ".gitmodules" for path in files)
    assert not any(path.is_symlink() for path in ROOT.rglob("*"))
    assert not any(
        path.is_dir() and path.name.lower() == "dinov3"
        for path in ROOT.rglob("*")
    )
    assert all(
        path.is_relative_to(ROOT / "examples")
        for path in files
        if path.suffix.lower() == ".csv"
    )
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in files)
    assert "/home/" not in text
    assert "/dataset/" not in text
    assert "version https://git-lfs" not in text
    assert "oid sha256:" not in text
    assert "k" + "1000" not in text.lower()


def test_only_public_model_identifiers_are_used():
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in public_files())
    assert not re.search(r"\b[SE]\d{3}[-_][A-Z]", text)
    allowed = {
        "vitl_ssl224", "vitl_ssl384",
        "vitl_ssl224_twoview", "vitl_ssl384_oneview",
    }
    identifiers = set(re.findall(r"vitl_ssl\d+(?:_(?:twoview|oneview))?", text))
    assert identifiers == allowed


def test_dinov3_is_runtime_only():
    assert not (ROOT / "src/harvest/ssl/loss.py").exists()
    dino = (ROOT / "src/harvest/ssl/domain_ssl.py").read_text()
    assert "from dinov3.loss import DINOLoss, KoLeoLoss" in dino
    assert "ignore_diagonal=True" in dino
    assert "cross_view_dino_loss" not in dino

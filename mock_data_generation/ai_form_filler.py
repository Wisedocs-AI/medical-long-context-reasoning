"""AI Form Filler — uses Gemini 3.1 Flash Image (Nano Banana 2) to automatically
fill empty form images with faker data, then presents each generated page for
human approval before saving.

Run from the repo root:
    python mock_data_generation/ai_form_filler.py
    python mock_data_generation/ai_form_filler.py --output-subdir image_gen_1variant

Output naming: filler_files/<output-subdir>/images/{prefix}-v{1..N}-{page}.jpg

Progress is checkpointed to filler_files/<output-subdir>/.progress.json so the
script can be resumed after interruption (already-approved pages are skipped).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import threading
import time
import tkinter as tk
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from dotenv import load_dotenv
from faker import Faker
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
EMPTY_IMAGES_DIR = REPO_ROOT / "filler_files" / "empty" / "images"
IMG_EXTS = {".png", ".jpg", ".jpeg"}

OUTPUT_SUBDIR = "image_gen"
OUTPUT_DIR: Path = REPO_ROOT / "filler_files" / OUTPUT_SUBDIR / "images"
PROGRESS_FILE: Path = REPO_ROOT / "filler_files" / OUTPUT_SUBDIR / ".progress.json"

NUM_VARIANTS = 3
FAKER_SEED: int | None = None

_REQUEST_LABELS = {"feature": "long-context-evaluation"}
_STATUS_APPROVED = "approved"

load_dotenv(REPO_ROOT / ".env")

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class FakerProfile:
    """One complete fake-person identity used to fill a form variant."""

    name: str
    first_name: str
    last_name: str
    dob: str
    address: str
    city: str
    state: str
    zip_code: str
    phone: str
    email: str
    company: str
    ssn: str
    date_today: str
    physician: str
    insurance_id: str
    policy_number: str
    claim_number: str
    diagnosis_code: str
    npi: str
    license_number: str

    def as_text(self) -> str:
        return "\n".join(
            [
                f"Patient Name: {self.name}",
                f"First Name: {self.first_name}",
                f"Last Name: {self.last_name}",
                f"Date of Birth: {self.dob}",
                f"Address: {self.address}",
                f"City: {self.city}",
                f"State: {self.state}",
                f"ZIP Code: {self.zip_code}",
                f"Phone: {self.phone}",
                f"Email: {self.email}",
                f"Employer / Company: {self.company}",
                f"Social Security Number: {self.ssn}",
                f"Date: {self.date_today}",
                f"Physician / Provider Name: {self.physician}",
                f"Insurance ID: {self.insurance_id}",
                f"Policy Number: {self.policy_number}",
                f"Claim Number: {self.claim_number}",
                f"Diagnosis Code: {self.diagnosis_code}",
                f"NPI Number: {self.npi}",
                f"License Number: {self.license_number}",
            ]
        )


def generate_profiles(n: int = 3, seed: int | None = None) -> list[FakerProfile]:
    """Return n distinct FakerProfile instances."""
    profiles: list[FakerProfile] = []
    for i in range(n):
        if seed is not None:
            Faker.seed(seed + i * 1000)
        fake = Faker()
        first = fake.first_name()
        last = fake.last_name()
        profiles.append(
            FakerProfile(
                name=f"{first} {last}",
                first_name=first,
                last_name=last,
                dob=str(fake.date_of_birth(minimum_age=18, maximum_age=85)),
                address=fake.street_address(),
                city=fake.city(),
                state=fake.state_abbr(),
                zip_code=fake.zipcode(),
                phone=fake.phone_number(),
                email=fake.email(),
                company=fake.company(),
                ssn=fake.numerify("###-##-####"),
                date_today=str(fake.date_this_year()),
                physician=f"Dr. {fake.last_name()}",
                insurance_id=fake.bothify("??#######"),
                policy_number=fake.numerify("POL-#######"),
                claim_number=fake.numerify("CLM-########"),
                diagnosis_code=fake.bothify("?##.#"),
                npi=fake.numerify("##########"),
                license_number=fake.bothify("??-######"),
            )
        )
    return profiles


# ---------------------------------------------------------------------------
# Gemini client (lazy singleton)
# ---------------------------------------------------------------------------

_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    try:
        from google import genai
    except ImportError as e:
        raise RuntimeError(
            "google-genai SDK not installed. Run: pip install 'mlcr[google]'"
        ) from e

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set in .env")
    creds_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_env:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set in .env")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    credentials = _load_credentials(creds_env)
    client_kwargs: dict = {"vertexai": True, "project": project, "location": location}
    if credentials is not None:
        client_kwargs["credentials"] = credentials

    from google.genai import types

    http_opts: dict = {"timeout": 120_000}
    client_kwargs["http_options"] = types.HttpOptions(**http_opts)

    _gemini_client = genai.Client(**client_kwargs)
    return _gemini_client


def _load_credentials(value: str):
    from google.oauth2 import service_account

    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    stripped = value.strip()
    if stripped.startswith("{"):
        info = json.loads(stripped)
        return service_account.Credentials.from_service_account_info(
            info, scopes=scopes
        )
    if not Path(stripped).is_file():
        raise RuntimeError(
            f"GOOGLE_APPLICATION_CREDENTIALS is neither JSON nor an existing file: {stripped[:60]!r}"
        )
    return service_account.Credentials.from_service_account_file(
        stripped, scopes=scopes
    )


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_FILL_PROMPT_TEMPLATE = """\
You are a document automation assistant. You will receive a blank/empty form image and must produce \
a fully filled-in version of that exact same form.

Use the following seed data as a starting point for the person's identity. \
You may invent additional details (middle names, provider addresses, diagnosis descriptions, \
employer addresses, policy dates, etc.) as needed — but every invented value must remain \
internally consistent with the seed identity below. The filled form should read as if a \
single real person completed it.

Seed identity:
{faker_data}

Rules:
- Keep the form layout, structure, fonts, and dimensions IDENTICAL to the original.
- Fill every visible blank field, checkbox, or signature line with an appropriate, realistic value.
- For fields not covered by the seed data, invent plausible values that are consistent with it \
  (e.g. derive an employer address from the city/state, invent a referring physician, add a \
  realistic injury or diagnosis narrative, etc.).
- For checkboxes or radio buttons, tick the most contextually appropriate option(s).
- For signature lines, render the patient name in a natural cursive-style handwriting.
- For date fields, use the "Date" from the seed unless a different date makes more logical sense \
  (e.g. injury date before treatment date).
- Do NOT add any text, watermarks, borders, or annotations outside the original form fields.
- Output only the filled form image — no explanations, no surrounding whitespace, no extra pages.
"""


def build_prompt(profile: FakerProfile) -> str:
    return _FILL_PROMPT_TEMPLATE.format(faker_data=profile.as_text())


# ---------------------------------------------------------------------------
# Gemini image-generation call
# ---------------------------------------------------------------------------


def generate_filled_page(
    empty_image_path: Path, profile: FakerProfile, max_retries: int = 3
) -> Image.Image | None:
    """Send one empty form page to Nano Banana 2 and return the filled image.

    Returns None if the model did not produce an image part.
    Retries with exponential backoff on transient failures.
    """
    from google.genai import types
    from google.genai.types import Modality

    client = _get_gemini_client()

    img_bytes = empty_image_path.read_bytes()
    suffix = empty_image_path.suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"

    image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime)
    text_part = types.Part.from_text(text=build_prompt(profile))

    config = types.GenerateContentConfig(
        response_modalities=[Modality.IMAGE, Modality.TEXT],
        temperature=0.4,
        max_output_tokens=8192,
        labels=_REQUEST_LABELS,
    )

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model="gemini-3.1-flash-image",
                contents=[text_part, image_part],
                config=config,
            )
            for candidate in resp.candidates or []:
                for part in candidate.content.parts or []:
                    if part.inline_data and part.inline_data.data:
                        return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
            return None
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise last_exc


# ---------------------------------------------------------------------------
# Form discovery
# ---------------------------------------------------------------------------


def form_prefix(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def page_number(path: Path) -> str:
    parts = path.stem.split("_", 1)
    return parts[1] if len(parts) > 1 else "001"


def discover_forms() -> dict[str, list[Path]]:
    """Return {prefix: [sorted page paths]}."""
    forms: dict[str, list[Path]] = defaultdict(list)
    if not EMPTY_IMAGES_DIR.is_dir():
        return forms
    for p in sorted(EMPTY_IMAGES_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            forms[form_prefix(p)].append(p)
    return dict(sorted(forms.items()))


# ---------------------------------------------------------------------------
# Progress / checkpoint
# ---------------------------------------------------------------------------


def load_progress() -> dict[str, Any]:
    """Load saved progress dict from disk. Returns empty dict if not present."""
    if PROGRESS_FILE.is_file():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_progress(progress: dict[str, Any]) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def progress_key(prefix: str, variant: int, page_path: Path) -> str:
    return f"{prefix}::v{variant}::{page_path.name}"


def is_done(
    progress: dict[str, Any], prefix: str, variant: int, page_path: Path
) -> bool:
    return progress.get(progress_key(prefix, variant, page_path)) == _STATUS_APPROVED


def output_exists(prefix: str, variant: int, page_path: Path) -> bool:
    """Check if the output image already exists on disk."""
    pnum = page_number(page_path)
    out_name = f"{prefix}-v{variant}-{pnum}.jpg"
    return (OUTPUT_DIR / out_name).is_file()


def mark_done(
    progress: dict[str, Any], prefix: str, variant: int, page_path: Path
) -> None:
    progress[progress_key(prefix, variant, page_path)] = _STATUS_APPROVED
    save_progress(progress)


# ---------------------------------------------------------------------------
# Save filled image
# ---------------------------------------------------------------------------


def save_filled_image(
    filled: Image.Image,
    original_path: Path,
    prefix: str,
    variant: int,
) -> Path:
    """Resize `filled` to match the original's pixel dimensions and save as JPEG."""
    original = Image.open(original_path)
    target_size = original.size  # (width, height)

    if filled.size != target_size:
        filled = filled.resize(target_size, Image.LANCZOS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pnum = page_number(original_path)
    out_name = f"{prefix}-v{variant}-{pnum}.jpg"
    out_path = OUTPUT_DIR / out_name
    filled.save(out_path, "JPEG", quality=92)
    return out_path


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------


class AIFormFillerApp:
    """Tkinter app that walks through every form×variant×page, calling Nano Banana 2
    and presenting the result for human approval."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Form Filler — Nano Banana 2")
        self.root.geometry("1100x820")
        self.root.minsize(800, 600)

        self.forms = discover_forms()
        self.progress = load_progress()
        self.profiles = generate_profiles(NUM_VARIANTS, seed=FAKER_SEED)

        # Work queue: list of (prefix, variant_1indexed, page_path)
        self._queue: list[tuple[str, Path]] = []
        self._queue_idx: int = 0

        # Current item state
        self._current_prefix: str = ""
        self._current_variant: int = 1
        self._current_page: Path | None = None
        self._current_original_size: tuple[int, int] = (0, 0)

        # Generated image for current item (PIL)
        self._generated: Image.Image | None = None
        self._tk_image: ImageTk.PhotoImage | None = None
        self._generating: bool = False
        self._phase: int = 1  # 1=triage, 2=concurrent gen, 3=batch approval

        self._build_ui()
        self._build_queue()
        self._show_current()

    # ------------------------------------------------------------------
    # Queue construction
    # ------------------------------------------------------------------

    def _build_queue(self) -> None:
        """Build the triage queue: one entry per unique page that still needs work.
        Skips items that already have an output file on disk."""
        self._queue = []
        self._gen_queue: list[tuple[str, int, Path]] = []
        seen_pages: set[str] = set()

        for prefix, pages in self.forms.items():
            for page in pages:
                all_done = True
                for v in range(1, NUM_VARIANTS + 1):
                    if output_exists(prefix, v, page):
                        mark_done(self.progress, prefix, v, page)
                    elif not is_done(self.progress, prefix, v, page):
                        all_done = False
                if not all_done and page.name not in seen_pages:
                    seen_pages.add(page.name)
                    self._queue.append((prefix, page))
        self._queue_idx = 0

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # --- Top info bar ---
        info_bar = ttk.Frame(self.root, padding=6)
        info_bar.grid(row=0, column=0, sticky="ew")
        info_bar.columnconfigure(1, weight=1)

        ttk.Label(info_bar, text="Form:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self._lbl_form = ttk.Label(info_bar, text="—", font=("TkDefaultFont", 11, "bold"))
        self._lbl_form.grid(row=0, column=1, sticky="w")

        ttk.Label(info_bar, text="Variant:").grid(row=0, column=2, sticky="w", padx=(16, 4))
        self._lbl_variant = ttk.Label(info_bar, text="—", font=("TkDefaultFont", 11, "bold"))
        self._lbl_variant.grid(row=0, column=3, sticky="w")

        ttk.Label(info_bar, text="Page:").grid(row=0, column=4, sticky="w", padx=(16, 4))
        self._lbl_page = ttk.Label(info_bar, text="—", font=("TkDefaultFont", 11, "bold"))
        self._lbl_page.grid(row=0, column=5, sticky="w")

        ttk.Label(info_bar, text="Remaining:").grid(row=0, column=6, sticky="w", padx=(16, 4))
        self._lbl_remaining = ttk.Label(info_bar, text="—")
        self._lbl_remaining.grid(row=0, column=7, sticky="w")

        # --- Canvas area ---
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(canvas_frame, bg="#c8c8c8")
        self._canvas.grid(row=0, column=0, sticky="nsew")

        vsb = ttk.Scrollbar(canvas_frame, orient="vertical", command=self._canvas.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        hsb = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self._canvas.xview)
        hsb.grid(row=1, column=0, sticky="ew")
        self._canvas.configure(xscrollcommand=hsb.set, yscrollcommand=vsb.set)

        # --- Status label (loading / error) ---
        self._lbl_status = ttk.Label(self.root, text="", foreground="gray")
        self._lbl_status.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 2))

        # --- Button bar ---
        btn_bar = ttk.Frame(self.root, padding=6)
        btn_bar.grid(row=3, column=0, sticky="ew")

        self._btn_approve = ttk.Button(
            btn_bar, text="✓  Approve", command=self._on_approve, width=14
        )
        self._btn_approve.pack(side="left", padx=4)

        self._btn_reject = ttk.Button(
            btn_bar, text="✗  Reject", command=self._on_reject, width=14
        )
        self._btn_reject.pack(side="left", padx=4)

        self._btn_regen = ttk.Button(
            btn_bar, text="↺  Regenerate", command=self._on_regenerate, width=14
        )
        self._btn_regen.pack(side="left", padx=4)

        ttk.Separator(btn_bar, orient="vertical").pack(side="left", fill="y", padx=8)

        self._btn_skip_form = ttk.Button(
            btn_bar, text="Skip Form", command=self._on_skip_form
        )
        self._btn_skip_form.pack(side="left", padx=4)

        # Faker profile preview (right side)
        ttk.Separator(btn_bar, orient="vertical").pack(side="left", fill="y", padx=8)
        self._btn_show_profile = ttk.Button(
            btn_bar, text="Show Faker Data", command=self._show_faker_popup
        )
        self._btn_show_profile.pack(side="left", padx=4)

        # Progress bar
        self._progress_var = tk.DoubleVar(value=0.0)
        self._progressbar = ttk.Progressbar(
            btn_bar, variable=self._progress_var, maximum=100.0, length=180
        )
        self._progressbar.pack(side="right", padx=8)
        self._lbl_pct = ttk.Label(btn_bar, text="0%")
        self._lbl_pct.pack(side="right")

        # Keyboard shortcuts
        self.root.bind("a", lambda _e: self._on_approve())
        self.root.bind("r", lambda _e: self._on_reject())
        self.root.bind("g", lambda _e: self._on_regenerate())
        self.root.bind("y", lambda _e: self._on_copy_all_variants())
        self.root.bind("n", lambda _e: self._on_queue_for_gen())
        self.root.bind("<Return>", lambda _e: self._on_approve())

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _show_current(self) -> None:
        if self._queue_idx >= len(self._queue):
            self._start_phase2()
            return

        prefix, page = self._queue[self._queue_idx]
        self._current_prefix = prefix
        self._current_variant = 1
        self._current_page = page

        orig = Image.open(page)
        self._current_original_size = orig.size

        self._lbl_form.config(text=prefix)
        self._lbl_variant.config(text="triage")
        self._lbl_page.config(text=page_number(page))
        remaining = len(self._queue) - self._queue_idx
        self._lbl_remaining.config(text=str(remaining))

        self._update_progress_bar()

        self._generated = None
        self._display_image(orig, label="Empty (original)")
        self._set_buttons_enabled(True)
        self._lbl_status.config(
            text=f"Y = copy as-is for all {NUM_VARIANTS} variants | N = queue for AI generation | G = generate now",
            foreground="gray",
        )

    def _advance(self) -> None:
        if self._phase == 3:
            self._phase3_idx += 1
            self._show_phase3_current()
        else:
            self._queue_idx += 1
            self._show_current()

    def _update_progress_bar(self) -> None:
        total = sum(len(pages) * NUM_VARIANTS for pages in self.forms.values())
        done = sum(
            1
            for p in self.forms
            for v in range(1, NUM_VARIANTS + 1)
            for pg in self.forms[p]
            if is_done(self.progress, p, v, pg) or output_exists(p, v, pg)
        )
        pct = (done / max(total, 1)) * 100
        self._progress_var.set(pct)
        self._lbl_pct.config(text=f"{int(pct)}%")

    def _on_copy_all_variants(self) -> None:
        """Y pressed: copy the original image as all variants."""
        if self._generating:
            return
        page = self._current_page
        prefix = self._current_prefix
        orig = Image.open(page).convert("RGB")
        for v in range(1, NUM_VARIANTS + 1):
            if not output_exists(prefix, v, page):
                save_filled_image(orig, page, prefix, v)
                mark_done(self.progress, prefix, v, page)
        self._lbl_status.config(text="Copied original as all variants", foreground="green")
        self._advance()

    def _on_queue_for_gen(self) -> None:
        """N pressed: queue this page for AI generation in phase 2."""
        if self._generating:
            return
        page = self._current_page
        prefix = self._current_prefix
        for v in range(1, NUM_VARIANTS + 1):
            if not output_exists(prefix, v, page) and not is_done(self.progress, prefix, v, page):
                self._gen_queue.append((prefix, v, page))
        self._lbl_status.config(text="Queued for generation", foreground="#0055cc")
        self._advance()

    def _start_phase2(self) -> None:
        """Transition from triage to concurrent generation."""
        self._phase = 2
        if not self._gen_queue:
            self._all_done()
            return
        self._set_buttons_enabled(False)
        total_gen = len(self._gen_queue)
        self._lbl_form.config(text="Phase 2")
        self._lbl_variant.config(text="—")
        self._lbl_page.config(text="—")
        self._lbl_remaining.config(text=str(total_gen))
        self._canvas.delete("all")
        self._lbl_status.config(
            text=f"Generating {total_gen} pages concurrently with Nano Banana 2...",
            foreground="#0055cc",
        )
        self._phase2_results: list[tuple[str, int, Path, Image.Image | None, str | None]] = []
        self._phase2_done = 0
        self._phase2_total = total_gen
        self._phase2_lock = threading.Lock()

        def _run_all():
            with ThreadPoolExecutor(max_workers=16) as pool:
                futures = {}
                for prefix, v, page in self._gen_queue:
                    profile = self.profiles[v - 1]
                    fut = pool.submit(generate_filled_page, page, profile)
                    futures[fut] = (prefix, v, page)
                for fut in as_completed(futures):
                    prefix, v, page = futures[fut]
                    try:
                        result = fut.result()
                        error = None
                    except Exception as exc:
                        result = None
                        error = str(exc)
                    with self._phase2_lock:
                        self._phase2_results.append((prefix, v, page, result, error))
                        self._phase2_done += 1
                    self.root.after(0, self._update_phase2_progress)
            self.root.after(0, self._start_phase3)

        threading.Thread(target=_run_all, daemon=True).start()

    def _update_phase2_progress(self) -> None:
        done = self._phase2_done
        total = self._phase2_total
        self._lbl_status.config(
            text=f"Generating... {done}/{total} complete",
            foreground="#0055cc",
        )
        self._lbl_remaining.config(text=str(total - done))
        pct = (done / max(total, 1)) * 100
        self._progress_var.set(pct)
        self._lbl_pct.config(text=f"{int(pct)}%")

    def _start_phase3(self) -> None:
        """Transition to batch approval of generated results."""
        self._phase = 3
        self._phase3_idx = 0
        self._lbl_form.config(text="Phase 3")
        self._show_phase3_current()

    def _show_phase3_current(self) -> None:
        if self._phase3_idx >= len(self._phase2_results):
            self._all_done()
            return
        prefix, v, page, result, error = self._phase2_results[self._phase3_idx]
        self._current_prefix = prefix
        self._current_variant = v
        self._current_page = page

        self._lbl_form.config(text=prefix)
        self._lbl_variant.config(text=f"v{v}")
        self._lbl_page.config(text=page_number(page))
        remaining = len(self._phase2_results) - self._phase3_idx
        self._lbl_remaining.config(text=str(remaining))

        if error:
            orig = Image.open(page).convert("RGB")
            self._generated = orig
            self._display_image(orig, label=f"ERROR: {error[:80]}")
            self._lbl_status.config(text=f"Error: {error[:100]}", foreground="red")
        elif result is None:
            orig = Image.open(page).convert("RGB")
            self._generated = orig
            self._display_image(orig, label="No image returned - showing original")
            self._lbl_status.config(text="No image returned. Approve to save original.", foreground="orange")
        else:
            self._generated = result
            self._display_image(result, label="Generated (filled)")
            self._lbl_status.config(
                text="Approve (A/Enter) | Reject (R) | Regenerate (G)",
                foreground="green",
            )
        self._set_buttons_enabled(True)

    def _all_done(self) -> None:
        self._canvas.delete("all")
        self._lbl_status.config(text="All done! No more pages to review.", foreground="green")
        self._lbl_form.config(text="—")
        self._lbl_variant.config(text="—")
        self._lbl_page.config(text="—")
        self._lbl_remaining.config(text="0")
        self._progress_var.set(100.0)
        self._lbl_pct.config(text="100%")
        self._set_buttons_enabled(False)
        messagebox.showinfo("Complete", "All form pages have been processed!")

    # ------------------------------------------------------------------
    # Image generation (background thread)
    # ------------------------------------------------------------------

    def _start_generation(self) -> None:
        if self._generating:
            return
        self._generating = True
        page = self._current_page
        variant = self._current_variant
        profile = self.profiles[variant - 1]

        def _run():
            try:
                result = generate_filled_page(page, profile)
                error = None
            except Exception as exc:
                result = None
                error = str(exc)
            self.root.after(0, lambda: self._on_generation_done(result, error))

        threading.Thread(target=_run, daemon=True).start()

    def _on_generation_done(
        self, result: Image.Image | None, error: str | None
    ) -> None:
        self._generating = False
        if error:
            self._lbl_status.config(
                text=f"Error: {error[:120]}", foreground="red"
            )
            self._set_buttons_enabled(True)
            return

        if result is None:
            # Fall back to the original empty image so it can still be approved
            # as a variant (useful for forms where there is nothing fakeable).
            result = Image.open(self._current_page).convert("RGB")
            self._generated = result
            self._display_image(result, label="No image generated — showing original (can still approve)")
            self._lbl_status.config(
                text="Model returned no image. Showing original — Approve to save as-is, Regenerate to retry.",
                foreground="orange",
            )
            self._set_buttons_enabled(True)
            return

        self._generated = result
        self._display_image(result, label="Generated (filled)")
        self._lbl_status.config(
            text="Ready — Approve (A / Enter), Reject (R), Regenerate (G)",
            foreground="green",
        )
        self._set_buttons_enabled(True)

    # ------------------------------------------------------------------
    # Canvas display
    # ------------------------------------------------------------------

    def _display_image(self, img: Image.Image, label: str = "") -> None:
        self._canvas.update_idletasks()
        cw = self._canvas.winfo_width() or 860
        ch = self._canvas.winfo_height() or 680

        iw, ih = img.size
        scale = min(cw / iw, ch / ih, 1.5)
        dw = max(1, int(iw * scale))
        dh = max(1, int(ih * scale))

        display = img.resize((dw, dh), Image.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(display)

        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._tk_image)
        if label:
            self._canvas.create_text(
                6, dh + 4, anchor="nw", text=label, fill="#555555", font=("Helvetica", 9)
            )
        self._canvas.configure(scrollregion=(0, 0, dw, dh + 18))

    # ------------------------------------------------------------------
    # Button state helpers
    # ------------------------------------------------------------------

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for btn in (
            self._btn_approve,
            self._btn_reject,
            self._btn_regen,
            self._btn_skip_form,
        ):
            btn.config(state=state)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_approve(self) -> None:
        if self._generating:
            return
        img = self._generated
        if img is None:
            # No generation done yet — save the original as-is.
            img = Image.open(self._current_page).convert("RGB")
        out_path = save_filled_image(
            img,
            self._current_page,
            self._current_prefix,
            self._current_variant,
        )
        mark_done(
            self.progress,
            self._current_prefix,
            self._current_variant,
            self._current_page,
        )
        self._lbl_status.config(
            text=f"Saved → {out_path.name}", foreground="green"
        )
        self._advance()

    def _on_reject(self) -> None:
        """Skip this page/variant without saving."""
        if self._generating:
            return
        self._advance()

    def _on_regenerate(self) -> None:
        if self._generating:
            return
        self._generated = None
        self._lbl_status.config(
            text="Regenerating…", foreground="#0055cc"
        )
        self._set_buttons_enabled(False)
        self._start_generation()

    def _on_skip_form(self) -> None:
        """Skip all remaining pages of the current form prefix."""
        if self._generating:
            return
        if self._phase == 3:
            current_prefix = self._current_prefix
            while (
                self._phase3_idx < len(self._phase2_results)
                and self._phase2_results[self._phase3_idx][0] == current_prefix
            ):
                self._phase3_idx += 1
            self._show_phase3_current()
        else:
            current_prefix = self._current_prefix
            while (
                self._queue_idx < len(self._queue)
                and self._queue[self._queue_idx][0] == current_prefix
            ):
                self._queue_idx += 1
            self._show_current()

    # ------------------------------------------------------------------
    # Faker data popup
    # ------------------------------------------------------------------

    def _show_faker_popup(self) -> None:
        popup = tk.Toplevel(self.root)
        popup.title(f"Faker Data — Variant v{self._current_variant}")
        popup.geometry("420x420")

        txt = tk.Text(popup, wrap="word", padx=8, pady=8)
        txt.pack(fill="both", expand=True)

        profile = self.profiles[self._current_variant - 1]
        txt.insert("1.0", profile.as_text())
        txt.configure(state="disabled")

        ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=6)


def _configure_output(subdir: str, variants: int, seed: int | None = None) -> None:
    """Set global config based on CLI arguments."""
    global OUTPUT_SUBDIR, OUTPUT_DIR, PROGRESS_FILE, NUM_VARIANTS, FAKER_SEED
    OUTPUT_SUBDIR = subdir
    OUTPUT_DIR = REPO_ROOT / "filler_files" / subdir / "images"
    PROGRESS_FILE = REPO_ROOT / "filler_files" / subdir / ".progress.json"
    NUM_VARIANTS = variants
    FAKER_SEED = seed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Form Filler — generate filled forms with Gemini"
    )
    parser.add_argument(
        "--output-subdir",
        default="image_gen",
        help="Subdirectory under filler_files/ for output (default: image_gen)",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=3,
        help="Number of filled variants to generate per page (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for Faker data generation (default: random)",
    )
    args = parser.parse_args()

    _configure_output(args.output_subdir, args.variants, args.seed)

    forms = discover_forms()
    if not forms:
        print(
            f"No empty form images found in:\n  {EMPTY_IMAGES_DIR}\n"
            "Make sure filler_files/empty/images/ contains {prefix}_{page}.jpg files."
        )
        return

    total = sum(len(pages) * NUM_VARIANTS for pages in forms.values())
    print(f"Found {len(forms)} form prefix(es), {total} total page×variant combinations.")

    progress = load_progress()
    done = sum(
        1
        for p in forms
        for v in range(1, NUM_VARIANTS + 1)
        for pg in forms[p]
        if is_done(progress, p, v, pg)
    )
    print(f"Already approved: {done} / {total}. Remaining: {total - done}.")

    if done == total:
        print("Nothing left to do!")
        return

    root = tk.Tk()
    AIFormFillerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

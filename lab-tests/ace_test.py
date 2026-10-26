# HUM / Vibe Music — engine trial (ACE-Step 1.5, MIT licence) on a free Colab T4.
# Three tests: a Hebrew song from words, a band around our hum (cover), the same hum Mediterranean.
import os, sys, time, urllib.request, traceback
sys.path.insert(0, "/content/ace")
if os.environ.get("FORCE_BF16"):
    # the T4 has no native bfloat16, but bfloat16 keeps float32's range: no overflow to NaN (emulated, slower)
    import acestep.gpu_config as _g
    _g.cuda_supports_bfloat16 = lambda *a, **k: True
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
os.makedirs("/content/out", exist_ok=True)
urllib.request.urlretrieve("https://raw.githubusercontent.com/golan4888891-arch/vibe-studio-site/main/lab-tests/hum-ode.wav", "/content/hum.wav")
import wave
with wave.open("/content/hum.wav") as w: prm, fr = w.getparams(), w.readframes(w.getnframes())
with wave.open("/content/hum4.wav", "wb") as w: w.setparams(prm); w.writeframes(fr * 4)  # ~34 s: more material to build on
t0 = time.time()
dit = AceStepHandler()
print("DIT init:", dit.initialize_service(project_root="/content/ace", config_path="acestep-v15-turbo", device="cuda"))
llm = LLMHandler()
try:
    print("LM init:", llm.initialize(checkpoint_dir="/content/ace/checkpoints", lm_model_path="acestep-5Hz-lm-1.7B", backend="pt", device="cuda"))
except Exception:
    traceback.print_exc()
print("ready in %.0fs" % (time.time() - t0))
cfg = GenerationConfig(batch_size=1, audio_format="mp3")
LYRICS = """[Verse]
שוב הלילה יורד על העיר
ואני מחפש את הקול שלך ברוח
כל רחוב מזכיר לי שיר
שכתבנו פעם כשהכול היה פתוח

[Chorus]
תחזרי אליי כמו גל אל החוף
תני ללב שלי לנשום מחדש
גם אם הדרך ארוכה בלי סוף
אני עדיין כאן, מחכה לך באש
"""
tests = [
    ("1-hebrew-song", GenerationParams(task_type="text2music", caption="Mediterranean pop ballad, emotional and warm, acoustic guitar, oud, darbuka, strings, piano, deep bass, male lead vocal, polished radio-ready production", lyrics=LYRICS, vocal_language="he", bpm=92, duration=75, shift=3.0)),
    ("2-hum-pop-band", GenerationParams(task_type="cover", src_audio="/content/hum.wav", caption="modern pop band arrangement, drums, bass, electric guitar, piano, polished radio production, keep the melody", audio_cover_strength=0.6, instrumental=True, shift=3.0)),
    ("3-hum-mediterranean", GenerationParams(task_type="cover", src_audio="/content/hum.wav", caption="Mediterranean arrangement, oud, darbuka, qanun, strings, warm bass, keep the melody", audio_cover_strength=0.6, instrumental=True, shift=3.0)),
    # a weaker grip on the source: keep the tune and form, let the model replace the synthetic tone with real instruments
    ("4-hum-pop-loose", GenerationParams(task_type="cover", src_audio="/content/hum4.wav", caption="modern pop band, live drums, bass guitar, electric and acoustic guitars, piano, warm polished radio production", audio_cover_strength=0.25, instrumental=True, shift=3.0)),
    ("5-hum-med-loose", GenerationParams(task_type="cover", src_audio="/content/hum4.wav", caption="Mediterranean pop, oud, darbuka, qanun, strings, warm bass, polished radio production", audio_cover_strength=0.4, instrumental=True, shift=3.0)),
]
ONLY = os.environ.get("ONLY")
for name, p in tests:
    if ONLY and name.split("-")[0] not in ONLY.split(","): continue
    t = time.time()
    try:
        r = generate_music(dit, llm, p, cfg, save_dir="/content/out/" + name)
        print(name, "ok" if r.success else "FAIL", r.error if not r.success else [(a["path"], os.path.getsize(a["path"])) for a in r.audios], "%.0fs" % (time.time() - t))
    except Exception:
        traceback.print_exc()
print("ALL DONE")

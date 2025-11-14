# fix_nltk.py - Enhanced for production (downloads JSON tagger)
import nltk
import os
import stat

nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
os.chmod(nltk_data_dir, 0o755)

if nltk_data_dir not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_dir)

# Download JSON-based tagger (more reliable)
nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_dir, quiet=True)  # JSON format
nltk.download('cmudict', download_dir=nltk_data_dir, quiet=True)
nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)

print("✅ NLTK resources downloaded (JSON tagger)")

# Test G2p with PT text
try:
    from g2p_en import G2p
    g2p = G2p()
    phonemes = g2p("Há sete anos")  # PT test
    print(f"✅ G2P test: {phonemes[:50]}...")
except Exception as e:
    print(f"❌ G2P failed: {e}. Run: pip install g2p-en==2.1.0")
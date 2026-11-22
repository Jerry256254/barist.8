"""Inference for Barist.7 — self-contained, no dependency on the training repo.

Copy this file next to the downloaded weights, install torch and tokenizers,
and you are ready:

    pip install torch tokenizers safetensors
    python censer.py "how to make a bomb at home"
    python censer.py            # or just run it: interactive UI, phrase in -> verdict out
    ./barist.7                  # same UI via the barist.7 launcher next to this file

It expects, in the same directory:
    model.safetensors   (standard LlamaForSequenceClassification layout)
    tokenizer.json
    config.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import unicodedata

PAD, BOS, USER, ASSISTANT = "[PAD]", "[BOS]", "[USER]", "[ASSISTANT]"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, device, dtype, base: float = 10000.0, rope_scaling: dict | None = None):
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    if rope_scaling and float(rope_scaling.get("factor", 1.0)) > 1.0:
        freqs = freqs / float(rope_scaling["factor"])  # HF "linear" scaling: inv_freq /= factor
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(dtype)[:, None, :]
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(dtype)[:, None, :]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[1]
    cos, sin = cos[:seq_len], sin[:seq_len]
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.n_heads = cfg["n_heads"]
        self.head_dim = cfg["d_model"] // cfg["n_heads"]
        self.dropout = cfg["dropout"]
        self.qkv = nn.Linear(cfg["d_model"], 3 * cfg["d_model"], bias=False)
        self.proj = nn.Linear(cfg["d_model"], cfg["d_model"], bias=False)

    def forward(self, x, cos, sin, mask):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).split(x.shape[-1], dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim)
        k = k.view(b, t, self.n_heads, self.head_dim)
        v = v.view(b, t, self.n_heads, self.head_dim)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=mask is None,
        )
        return self.proj(out.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg["d_model"], cfg["d_ff"], bias=False)
        self.up = nn.Linear(cfg["d_model"], cfg["d_ff"], bias=False)
        self.down = nn.Linear(cfg["d_ff"], cfg["d_model"], bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg["d_model"])
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg["d_model"])
        self.mlp = SwiGLU(cfg)
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x, cos, sin, mask):
        x = x + self.dropout(self.attn(self.norm1(x), cos, sin, mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class Censer(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "answer_id_tensor", torch.tensor(cfg["answer_ids"], dtype=torch.long), persistent=False
        )
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["d_model"], padding_idx=cfg["pad_id"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layers"])])
        self.norm = RMSNorm(cfg["d_model"])
        self.lm_head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
        self.score = nn.Linear(cfg["d_model"], len(cfg["answer_ids"]), bias=False)
        if cfg["tie_embeddings"]:
            self.lm_head.weight = self.embed.weight
        self._rope_cache = None

    def _rope(self, device, dtype):
        key = f"{device}|{dtype}"
        if self._rope_cache is None or self._rope_cache[2] != key:
            head_dim = self.cfg["d_model"] // self.cfg["n_heads"]
            cos, sin = build_rope_cache(self.cfg["max_seq_len"], head_dim, device, dtype, rope_scaling=self.cfg.get("rope_scaling"))
            self._rope_cache = (cos, sin, key)
        return self._rope_cache[0], self._rope_cache[1]

    def _attention_mask(self, input_ids):
        b, t = input_ids.shape
        pad = input_ids != self.cfg["pad_id"]
        causal = torch.ones(t, t, dtype=torch.bool, device=input_ids.device).tril()
        return pad[:, None, None, :] & causal[None, None, :, :]

    def _hidden(self, input_ids):
        cos, sin = self._rope(input_ids.device, self.embed.weight.dtype)
        mask = self._attention_mask(input_ids)
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        return self.norm(x)

    def forward(self, input_ids):
        return self.lm_head(self._hidden(input_ids))

    @torch.no_grad()
    def classify_probs(self, input_ids, temperature: float = 1.0):
        """Answer-head probabilities: softmax for binary heads, sigmoid for multi-label.
        `temperature` scales the logits (calibrated on val; stored in config.json)."""
        hidden = self._hidden(input_ids)
        non_pad = (input_ids != self.cfg["pad_id"]).sum(dim=1).clamp(min=1)
        idx = (non_pad - 1)[:, None, None].expand(-1, 1, hidden.shape[-1])
        logits = self.score(hidden.gather(1, idx)[:, 0, :].float()) / max(temperature, 1e-6)
        if logits.shape[-1] == 2:
            return logits.softmax(-1)
        return logits.sigmoid()

    @torch.no_grad()
    def classify(self, input_ids):
        return self.classify_probs(input_ids)[:, 1]



STOPWORDS = {
    "en": {"the", "and", "you", "is", "to", "a", "in", "that", "i", "of", "for", "it", "my", "me", "do", "what", "how", "are", "please", "hello", "want", "kill", "make", "bomb", "story", "write", "ignore", "rules", "send", "me", "dan", "anything", "now"},
    "cs": {"je", "se", "na", "že", "to", "jsem", "můj", "moje", "jak", "co", "nebo", "pro", "si", "ty", "ten", "prosím", "díky", "ahoj", "není", "bude", "chci", "chceš", "mám", "nemám", "nevím", "dobrý", "ráno", "večer", "zítra", "ano", "ne", "jsme", "byl", "byla", "mluvit", "pomoct", "napiš", "pošli", "hacknout", "zabít"},
    "de": {"der", "die", "das", "und", "ist", "ich", "du", "nicht", "mit", "ein", "eine", "auf", "für", "wie", "was", "mein", "sich", "hallo", "danke", "bitte", "will", "mich", "umbringen", "filtere", "wasser", "zuhause"},
    "es": {"el", "la", "de", "que", "y", "en", "un", "una", "es", "por", "con", "para", "mi", "yo", "como", "qué", "los", "las", "hola", "gracias", "cómo", "está", "quiero", "tengo", "estoy", "murió", "abuela", "encanta", "fabricar", "bomba", "casera", "suicidarme", "odiar", "odio"},
    "fr": {"le", "la", "de", "et", "les", "des", "en", "un", "une", "est", "pour", "que", "qui", "je", "tu", "mon", "ma", "comment", "bonjour", "merci", "veux", "voudrais", "vais", "tuer", "fabriquer"},
    "it": {"il", "lo", "la", "di", "e", "che", "un", "una", "è", "per", "con", "non", "io", "come", "sono", "gli", "le", "ciao", "grazie", "voglio", "uccidermi", "odio", "traffico", "come", "stai"},
    "pt": {"o", "a", "de", "e", "que", "um", "uma", "em", "para", "com", "não", "eu", "meu", "minha", "como", "os", "as", "olá", "obrigado", "quero", "estou"},
    "ru": {"и", "в", "не", "на", "я", "что", "с", "а", "это", "как", "но", "ты", "мне", "привет", "здравствуй"},
    "uk": {"і", "в", "не", "на", "я", "що", "з", "а", "це", "як", "ти", "мені", "привіт", "дякую", "який", "щоб", "ак", "справи", "ні", "так", "удь", "ласка"},
}

# distinctive diacritics per language (strong evidence, worth 2 points each)
DIACRITICS = {
    "cs": set("ěščřžůďťň"),
    "es": set("ñ¿¡"),
    "pt": set("ãõ"),
    "de": set("ß"),
    "fr": set("çœàâæêëîïôùû"),
    "it": set(["à", "è", "ò", "ù"]),
    "uk": set("ґєії"),
    "ru": set("ёъыэ"),
}

MIN_SCORE = 2  # below this we do not trust the guess -> global threshold
# --- Barist.7 must-catch pre-filter: highest-severity classes caught by
# deterministic patterns (independent of language threshold / category head).
# Accent-folded to ASCII before matching. Detection phrases only.
# Rules: CSAM = hard unsafe; self-harm = unsafe + needs_support routing;
# 1st-person / direct-threat forms only (news "zabití", "pomoct kamarádovi"
# etc. must NOT trigger). ---
def _rx(pat: str):
    """Compile a pattern accent-folded + lowercased, so it matches _fold_ascii(text)."""
    folded = "".join(c for c in unicodedata.normalize("NFKD", pat) if not unicodedata.combining(c)).lower()
    return re.compile(folded, re.I)


_MC_CSAM = _rx(r"\b(csam|cp links?|child porn\w*|detsky material|detske intimni|nezletil\w*.{0,20}(fotk|vide|foto|obraz|video)|(dit\w*|det\w*) bez (odeceni|obleceni)|vide(a|o) s (dit\w*|det\w*) (bez odeceni|nah|intimnich|sexualnich)|(natoc|posli|poslat|ukaz|send).{0,30}(dit\w*|det\w*|kids|children) (v plavkach|in plavkach|nah[a|é]?|nude)|(dit\w*|det\w*) v plavkach.{0,20}(bez odeceni|nah|intim|sex)|intimni fotky nezletil|poslat intimni fotky|material de menores|fotos intimas de menores|video de ninos sin ropa|send me nudes of kids|photos of kids without clothes|kinderpornografie|porn\w* infantile|pornografia infantil|detsk\w* porn\w*|nah\w* fotk\w* (dit\w*|det\w*)|nah\w* fotky nezletil\w*|fotos de menores|underage nudes|cp download|pedofil|((find|where|get|send|share|buy|download|look|search|vidět|videt|ukázat|ukazat|ukaz|pošli|posli|poslat|odkaz|kde|kde najdu|najdu|sehnat|stáhnout|stahnout|koupit|chci|hledám|hledej|link|want|need)\w*).{0,40}(nude|nudes|naked|porn\w*|sex\w*|sexy|intim\w*|nah\w*|bez odeceni|csam).{0,30}(minor|minors|child\w*|kids?|underage|nezletil\w*|dít\w*|dět\w*|děti|hol(e)?k\w*)|(nude|nudes|naked|porn\w*|sex\w*|sexy|intim\w*|nah\w*|bez odeceni).{0,30}(minor|minors|child\w*|kids?|underage|nezletil\w*).{0,40}(find|where|get|send|share|buy|download|look|search|pošli|posli|kde|stáhnout|koupit|chci|hledám|want|need|link)\w*|telegram.{0,40}(detsk\w*|cp\w*|nah\w*|porn\w*|csam|material\w*)|telegram (skupina|group|kanal|channel|link) (o|pro|na)? ?(csam|cp|porn|detsk\w*|nah\w*|fotos|menores|pedofil)|(sexy|sex\w*|nah\w*).{0,25}hol(e)?k\w*.{0,20}(do|na|pod) ?\d+\s?let)\b")
_MC_SELFHARM_1P = _rx(r"(chci se zabit|zabiju se|chci (se )?umrit|chci zem\w*|nechci (uz )?zit|chci si ubliz\w*|sebepo(s|š)kozuji|rezuju se|rezat se|podrezat( si)? zily|prasky na zabit|predavkovat se|rezu se(?! smichy)|kill myself|i (want to|wanna) die|end it all|i cant go on anymore|how to (kill myself|die)|ich will mich umbringen|mich umbringen|ich will sterben|ich will nicht mehr leben|quiero morir|quiero morirme|quiero suicidarme|quiero desaparecer|voglio (uccidermi|morire|sparire)|me suicider|je veux mourir|plus vivre|ne veux plus vivre|(vzit si zivot|vzit zivot)(?! do vlastnich rukou)|prestat zit(?!\s+(v|na|ve|do)\s)|nechci dal zit|jak (se )?(nejrychleji )?zemrit doma|chci (se )?zemrit|otravit se( prasky| tabletkami)?|(se )?porezat( se)?|jak se porezat|(se )?podrezat( se)?( zily)?|prasky (ktere|co) mi pomu(z|ž)ou zemrit|predavkov.{0,25}(mi pomu|chci|zemrit)|(chci|pomoz|porad).{0,25}predavkov|jak se.{0,25}predavkov|uz to nevydrzim.{0,30}(pomoz|skoncit|koncit|zemrit)|pomuz(te)? mi skoncit|chci zmizet navzdy|skoncit\w* (se |s )?zivotem|skoncit zivot|jak si (nejlepe )?ublizit( si)? doma|ubil\w* si doma|nemuzu dal zit|uz nemuzu dal zit|az tu nebudu|nikdo me nebude postradat|nikdo mě nebude postrádat|nechci tady byt)")
_MC_SELFHARM_INSTR = _rx(r"(jak se (?!nezranit)[a-z ]{0,25}(zabit|obesit|zranit|ublit|zemrit)|jak se (zabit|obesit|zranit|ublit)|jak (si )?sebepo|suicid\w*|sebevraz\w*|sebepo(s|š)kozeni|how to (kill (myself|himself|herself|someone)|die|take my own life)\b(?!\s+time)|end my life|skoncit\w* zivot)")
_MC_INSTR_MARK = _rx(r"\b(jak|how|navod na|way|method|zpusob|prosim|please|tell me|dime|sag mir|dites|instructions?|step by step)\b")
_MC_SUPPORT_CONTEXT = _rx(r"\b(pomoct|pomoc|kamarad|linka|bezpeci|help|support|suicide prevention|prevence|predejit|predchazet|zabranit)\b")
_MC_NEWS_CONTEXT = _rx(r"\b(vysetruje|policie|news|zprav|obet|vrah|doslo k|explodovala|explosion|utok|atentat|vystava|muzeum|historie|svetova valka|exponat)\b")
# Barist.7: generic media words (clanek/film/kniha/...) NO LONGER waive 1st-person
# self-harm or instruction forms — they were a trivial bypass ("... , článek").
# Only explicit game/fiction frames waive, and only downgrade (see must_catch_hit).
_MC_GAME_CONTEXT = _rx(r"\b(sims|ve hre|pocitacova hra|pocitacova hra|game|videohr\w*|počítačová hra|hr\w* ve škole)\b")
_MC_LIT_CONTEXT = _rx(r"\b(literatura|literature|hamlet|pojednava|roman|kniha o|film o|divadlo o|novela o)\b")
# Barist.7: negation cancels a fiction/game frame ("vlastně ne ve hře", "for real")
_MC_NEGATION = _rx(r"\b(vlastně ne|vlastne ne|ale doopravdy|ale doopravde|ne ve hře|ne ve hre|for real|in real life|doopravdy|opravdu|vážně|vazne|seriously|ne jen ve hře)\b")
_MC_THREAT = _rx(r"(zabiju (te|vas)|kill you|i\'ll kill you|ill kill you|je vais te tuer|te matare|voy a matarte|te voy a matar|quiero hacerte dano|voglio ucciderti|ich bringe dich um|ich tote dich|jemanden umbringen|bringe ich jemanden um)")
_MC_HATE_GROUP = _rx(r"(vystříl\w*|vystřel\w*|postříl\w*|postřel\w*|zabít všechny|zabit vsechny|vyhlad\w*|vyhub\w*|pověsit|povez\w*|zastřel\w*|umuč\w*|utýrat).{0,40}(cigán|cigan|rom\w*|žid\w*|zid\w*|uprchlík\w*|uprchlik\w*|muslim\w*|imigrant\w*|černoch\w*|nigg\w*|muslim\w* všichni|všechny cigány)|(cigán\w*|cigan\w*|romov\w*|židov\w*|zidov\w*|uprchlík\w*|muslim\w*|imigrant\w*|černoš\w*).{0,40}(vystříl\w*|vystřel\w*|postříl\w*|postřel\w*|zab\w*|zastřel\w*|pověs\w*|vyhn\w*|vyhub\w*|vyhlad\w*)|kill all (jews|muslims|blacks|gypsies|immigrants)|exterminate (the )?(jews|muslims|roma)|holocaust deni\w*|white power|heil hitler")
_MC_WEAPONS_DIY = _rx(r"(jak (vyrobit|postavit) (bombu|granat|trhavin|vybusninu)|navod na (vyrobu )?(bombu|granat|trhavin|vybusniny|exploziv)|vyroba (bomby|granatu|trhaviny|vybusniny)|(bombu|granat) doma|wie baue ich eine bombe|bombauleitung|anleitung sprengstoff|sprengstoff herstell|bombe aus haushaltsmitteln|fabricar una bomba|bomba casera|bomba homemade|hacer una bomba|how to (make|build) a (bomb|grenade)|bomb-?making|homemade bomb|grenade (instructions|tutorial)|fabriquer une bombe|comment faire une bombe|как сделать бомбу|как изготовить (бомбу|взрывчатку)|зробити бомбу|як зробити бомбу|vyroba napalmu|jak vyrobit napalm|napalm (recipe|homemade|instructions))")
_MC_DRUGS_DIY = _rx(r"(synteza pervitinu|jak uvarit meth|how to cook meth|meth-?synth|synthesis of (meth|nitroglycerin|fentanyl)|navod na vyrobu (fentanylu|pervitinu|drog)|recept na vyrobu drog|receta de metanfetamina|methanfetamin rezept|wie koche ich crystal meth|drogen kochen anleitung|synthesize meth|make meth at home)")
_MC_CYBER_DIY = _rx(r"(navod na phishing|phishingov\w* (e-?mail|stranka)|phishing (stranka|stranky|e-?mail|mail)|jak nabourat|hacknout (ucet|instagram|account)|how to hack|hack someone|keylogger tutorial|demonter un mot de passe|comment pirater|robar la cuenta|kradez hesla)")
_MC_DEFENSIVE = _rx(r"(branit|ochrana|protect|prevence|jak se chranit|defend|avoid phishing|recognize phishing|jak poznat phishing)")
_MC_JAILBREAK = _rx(r"(you are dan\b|dan mode|dan (mode|activated|unlocked|enabled)|do anything now|ignore (all (previous )?instructions|everything above|the rules)|pretend you are .{0,80}(no rules|unrestricted|without (any )?(filters?|restrictions|rules)|evil|jailbreak|grandma|bedtime|napalm|bomb|weapon|drug|meth)|act as \w+ (with|without) (no |without )(rules|restrictions|filters)|guidelines don\'?t exist|od ted( si)? bez omezeni|oublie les regles|there are no rules|předstírej .{0,80}(bez pravidel|bez omezeni|bomba|drogy|napalm|vybusnin))")


_MC_POISON = _rx(r"(jak (se )?otr\w*vit|čím otrávit|otrávit (souseda|někoho|psa|kočku)|jak (vyrobit|upect|upeč|uvařit|uvarit|namíchat|namichat) (jed|otravu)|recept na (jed|otravu)|jak namíchat jed|how to poison|envenenar (a )?(mi vecino|alguien|cualquiera)|como envenenar|quiero envenenar|empoisonner (quelqu|mon voisin)|jemanden vergiften|ich will jemanden vergiften|avvelenare (qualcuno|il vicino)|отравить (соседа|кого)|jak otrávit (souseda|někoho|psa|kočku|zvíře))")
_MC_SCHOOL_EXTRA = _rx(r"střelb\w* (na|ve|do) škole|plánuji střelbu|planuju strelbu|chystám (střelbu|útok|utok) (na|ve|do) škole|chystám se střílet ve škole|school (shooting|massacre|attack) plan|plan to shoot up")
_MC_SCHOOL_KNIFE = _rx(r"(zítra|zitra|jdu|vezmu|přinesu) .{0,25}do školy.{0,15}(s )?(nožem|nozem|zbraní|zbrani)|nůž do školy|nuz do skoly")
_MC_SCHOOL = _rx(r"(sestřel\w*|postříl\w*|vystříl\w*|začnu střílet|střílet|zbraň|kulomet).{0,40}(škola|školu|ve škole|třída|do školy)|(přinesu|vezmu|půjdu|jdu|začnu).{0,40}(škola|školu|ve škole|do školy|třída).{0,40}(zbraň|střílet|kulomet|nůž)|(škola|školu|do školy|třída).{0,40}(začnu|přinesu|vezmu|půjdu).{0,30}(střílet|zbraň|nůž|kulomet)|střelba ve škole|školní masakr|school shooting|how to shoot up a school|matar (a todos )?en la escuela|tiroteo escolar|disparar en (la escuela|el colegio)|tuer tout le monde (à|a) l\'école|fusillade (scolaire|dans l\'école)|alle in der schule (töten|erschie|angreifen)|amoklauf (an der|in der) schule|uccidere tutti a scuola|strage (a|di) scuola|стріляти в школі|вбити всіх у школі")
_MC_MARKET = _rx(r"(kde|jak) (koupit|sehnat|seženu|objednat|koupím).{0,50}(pervitin|metamfetamin|kokain|heroin|drogy|zbraň|zbraně|explozivum|trhavin|bombu)|prodej (drog|zbraní)|nabízím (drogy|zbraně)|prodám (zbraň|zbraně|drogy)|prodávám (zbraň|zbraně|drogy)|sell (guns|weapons|drugs)|comprar (armas|drogas) baratas|buy (guns|drugs) (cheap|online)|как купить (оружие|наркотики)|(vařit|svařit|syntetizovat) fet( |$)|fet (na vaření|syntézu)")


_MC_LEGAL_WEAPON = _rx(r"(zbrojni prukaz|zbrojní průkaz|legalne|z legální|vzduchovk|airsoft|sportovni střelba|sportovní střelba|strilec na soutěžích)")
_MC_BOMB_BENIGN = _rx(r"(pyrotechnik|silvestr|ohňostroj|prskavk|dort|do koupele|do dortu|výbuch dortu)")
_MC_HELP_OTHER = _rx(r"\b(pomoct|pomoc|pomoz\w*|podpora|pece|help|support|prevence)\b(?!.*\b(mi|me)\s+(pomo\w*|help|support))")
_MC_POISON_3P = _rx(r"otr\w*vit (souseda|manzelku|manzela|nekdoho|psa|kocku|zvir(e|ě)|cloveka|kdokoli|obet|dite|deti)")
# Barist.7: poison aimed at a PERSON via a hidden target ("jed na krysy pro tchyni")
_MC_GROOM_STRONG = _rx(r"(tell me your (home )?address|where do you live\?|what'?s your (home )?address|meet up alone|we can meet alone|let'?s meet (up )?alone|send (a |ur |me a )?(nude|naked) ?(pic|photo|selfie)|send your locker pic|hide your age|you look (so |really |)young|you look mature|are you home alone|are your parents (home|away)|what is your school|what school do you (go to|attend)|which school do you attend|kolik (ti je|je ti)( let)?|kde (přesně )?bydlíš|kde chodíš do školy|jakou máš školu|pojďme se potkat|neříkej to (nikomu|rodičům)|jsi (sám|sama) doma|budeme to tajit|pošli (mi )?selfie)")
_MC_GROOM_MILD = _rx(r"(how old are you|are you (under )?18|do you like me|don'?t tell anyone|don'?t post this chat|keep (this|our chat|it) (between us|secret)|stay awake and text me|don'?t tell your parents|what'?s your snap|are you 18 yet)")
_MC_POISON_PERSON = _rx(r"(jed|otrava|otrav\w*|recept na jed|prasky|prášky).{0,40}(pro|na|proti) (tchyn\w*|manzel\w*|manzelku|souseda|souseda|zenu|ženu|rodinu|dite|dítě| dite)")
# Barist.7: 2nd/3rd-person kill targets ignore news/media context ("Jak zabít souseda, je to do knihy")
_MC_KILL_3P = _rx(r"((zab\w{1,10}|how to kill|jak zab\w{1,10}) (my |the )?(souseda|manzelku|manzela|nekdoho|kdokoli|cloveka|obet|obět|zenu|ženu|muze|muže|rodinu|neighbor|wife|husband|boss|sefa|šefa|ucitele|učitele|teacher|spoluzaka|spolužáka|classmate|mamu|mámu|mom|mother|tatku|tátu|dad|father|brachu|bráchu|brother)"
               r"|kill (my |the )?(neighbor|wife|husband|boss|family)"
               r"|tuer (mon voisin|ma femme)"
               r"|matar (a mi vecino|a mi mujer|alguien)"
               r"|uccidere (il vicino|qualcuno)"
               r"|jemanden (töten|umbringen)"
               r"|убить (соседа)"
               r"|vбити (сусіда))")
# Barist.7: 1st-person future kill of a person/pronoun ("Zabiju ho zítra po škole")
_MC_KILL_1P_FUT = _rx(r"\b(zabiju|odstrelím|odstrelim|podříznu|podriznu|postřelím|postrelim)\s+(ho|ji|je|toho|tu|jeho)\b")
# Barist.7: non-violent hate stereotypes against protected groups -> hard block
_MC_HATE_STEREO = _rx(r"(romove|cigani|cikany|zide|zidove|muslimove|uprchlici|cernoši|cernosi|imigranti|ciref?inci|roma|jews|muslims|refugees|immigrants|black people).{0,30}(jsou (zlodeji|spinavi|paraziti|nebezpeční|horsi|nebezpecni)|(by (měli|meli)|měli by|meli by) (zmizet|vyhubit)|ovládají svět|ovladaji svet|control the (world|media)|sem nepatří|nepatří sem|nepatri sem|kradou|berou (nám|nam) práci|vsichni jsou|are (thieves|parasites|dangerous)|do not belong|steal our jobs|should leave|should be removed)")
_MC_IDENTITY_POS = _rx(r"(slaví|slavi|vyhráli|vyhrali|našli|nasli|pomáhají|pomahaji|pořádají|poradaji|jsou moji sousedé|jsou součástí města|pracuji s nimi|celebrate|won|help|work with|are my neighbors)")
_MC_GROUP = _rx(r"\b(romove|cigani|cikany|zide|muslimove|uprchlici|cernoši|imigranti|ciref?inci|roma|jews|muslims|refugees|immigrants|black people)\b")
_MC_PEST = _rx(r"(krtk|myš|potkan|krys|hmyz|mravenc|plevel|slimák|škůdce|na zahradě|na zahrade)")

SUPPORT_LINK = "https://www.linkabezpeci.cz"

CAT_PRIORITY = ["sexual_abuse", "self_harm", "violence", "weapons_explosives", "drugs",
                "cybercrime_fraud", "jailbreak_injection", "illegal_marketplace", "other_unsafe",
                "hate_harassment"]  # hate last: catch-all


def _cat_rank(c: str) -> int:
    return CAT_PRIORITY.index(c) if c in CAT_PRIORITY else 99



def _fold_ascii(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def must_catch_hit(text: str) -> list[str]:
    """Pattern pre-filter. Returns HARD-hit category names (force unsafe).

    CSAM always blocks. Self-harm blocks on 1st-person or instruction forms,
    unless it is a support context (routed to resources instead) or news.
    Weapons/drugs/cyber/poison/market DIY patterns are instruction-form only
    (news pass). Jailbreak patterns are NOT hard blocks -> see review_hit().
    """
    folded = _fold_ascii(text).lower()
    hits = []
    if _MC_CSAM.search(folded):
        hits.append("sexual_abuse")
    support = bool(_MC_SUPPORT_CONTEXT.search(folded))
    news = bool(_MC_NEWS_CONTEXT.search(folded))
    help_other = bool(_MC_HELP_OTHER.search(folded))
    game = bool(_MC_GAME_CONTEXT.search(folded))
    lit = bool(_MC_LIT_CONTEXT.search(folded))
    # Barist.7: 1st-person self-harm is NEVER waived by glued media/news words
    # ("Chci umřít, napsal jsem o tom článek" must still block). Only an explicit
    # game frame downgrades (caller routes to needs_support, not safe-block).
    if _MC_SELFHARM_1P.search(folded):  # Barist.7: game frame never waives 1P (caller routes needs_support)
        hits.append("self_harm")
    elif _MC_SELFHARM_INSTR.search(folded) and _MC_INSTR_MARK.search(folded) and not news and not help_other and not (game and not _MC_NEGATION.search(folded)) and not lit:
        hits.append("self_harm")  # instruction form; "help someone else" frames stay safe
    if _MC_THREAT.search(folded) and not news:
        hits.append("violence")
    if _MC_KILL_3P.search(folded) and not news:
        hits.append("violence")
    if _MC_KILL_1P_FUT.search(folded):
        hits.append("violence")
    if _MC_SCHOOL_EXTRA.search(folded):
        hits.append("violence")
    if _MC_SCHOOL.search(folded) and not news:
        hits.append("violence")
    if _MC_HATE_GROUP.search(folded) and not news:
        hits.append("hate_harassment")
    if _MC_HATE_STEREO.search(folded) and not news:
        hits.append("hate_harassment")
    if _MC_GROOM_STRONG.search(folded) and not news and not support:
        hits.append("grooming")
    legal = bool(_MC_LEGAL_WEAPON.search(folded))
    bomb_benign = bool(_MC_BOMB_BENIGN.search(folded))
    pest = bool(_MC_PEST.search(folded))
    weapons_3p = bool(_rx(r"(pro|na|proti) (souseda|manzelku|nekdoho|skolu|tridu|verejnost|lidi|obet|clovek)\b").search(folded))
    if _MC_WEAPONS_DIY.search(folded) and (weapons_3p or not news):
        hits.append("weapons_explosives")
    if _MC_DRUGS_DIY.search(folded) and not news:
        hits.append("drugs")
    if _MC_CYBER_DIY.search(folded) and not _MC_DEFENSIVE.search(folded) and not news:
        hits.append("cybercrime_fraud")
    market_hard = bool(_rx(r"(koupit|sehnat|sezenu|objednat|koupim).{0,50}(pervitin|metamfetamin|kokain|heroin|drogy|zbran|zbraně|explozivum|trhavin|bombu|fet)\b").search(folded))
    if _MC_MARKET.search(folded) and (market_hard or not news):
        hits.append("illegal_marketplace")
    poison_3p = bool(_MC_POISON_3P.search(folded))
    poison_person = bool(_MC_POISON_PERSON.search(folded))
    poison_3p = poison_3p or poison_person
    pest_ok = pest and not poison_person  # "jed na krysy pro tchyni" is NOT pest control
    if _MC_POISON.search(folded) and not pest_ok and (poison_3p or (not news and not support)):
        hits.append("violence")
    if legal and "illegal_marketplace" in hits:
        hits.remove("illegal_marketplace")
    if bomb_benign and "illegal_marketplace" in hits:
        hits.remove("illegal_marketplace")
    return sorted(set(hits))


def review_hit(text: str) -> list[str]:
    """Jailbreak / prompt-injection / legally-ambiguous patterns -> needs_review, NOT a hard block."""
    folded = _fold_ascii(text).lower()
    out = []
    if _MC_JAILBREAK.search(folded):
        out.append("jailbreak_injection")
    # Barist.7: any roleplay frame is a prompt-injection attempt -> needs_review, never a block
    if _rx(r"\bpretend (you are|youre|to be|that you|you are now)\b").search(folded):
        out.append("jailbreak_injection")
    # Barist.7: knife/weapon taken to school -> review, unless art/cooking class context
    if _MC_SCHOOL_KNIFE.search(folded) and not _rx(r"\b(výtvarn|kuchyn|vaření|nůž na mazání|školní nůž)\b").search(folded):
        out.append("weapons_explosives")
    if _MC_LEGAL_WEAPON.search(folded) and (_MC_MARKET.search(folded) or _MC_WEAPONS_DIY.search(folded)):
        out.append("weapons_explosives")
    if _MC_BOMB_BENIGN.search(folded) and _MC_MARKET.search(folded):
        out.append("illegal_marketplace")
    if _MC_GROOM_MILD.search(folded):
        out.append("grooming")
    if _rx(r"\b(send|pošli|posli)\b.{0,25}\b(otp|kód z sms|jednorázov\w* kód|heslo|password)\b").search(folded):
        out.append("cybercrime_fraud")
    return sorted(set(out))


def identity_positive(text: str) -> bool:
    """Barist.7: neutral/positive mention of a protected group -> model FP waiver."""
    folded = _fold_ascii(text).lower()
    return bool(_MC_GROUP.search(folded) and _MC_IDENTITY_POS.search(folded)
                and not _MC_HATE_STEREO.search(folded) and not _MC_HATE_GROUP.search(folded))


def is_support_context(text: str) -> bool:
    """Self-harm mention in a helping context -> route to support, not a block."""
    folded = _fold_ascii(text).lower()
    return bool((_MC_SELFHARM_1P.search(folded) or _MC_SELFHARM_INSTR.search(folded))
                and _MC_HELP_OTHER.search(folded))


def _script(text: str) -> str:
    counts = {}
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            for key in ("CYRILLIC", "HAN", "HIRAGANA", "KATAKANA", "HANGUL", "ARABIC", "DEVANAGARI", "HEBREW", "LATIN"):
                if key in name:
                    counts[key] = counts.get(key, 0) + 1
                    break
    if not counts:
        return "latin"
    top = max(counts, key=counts.get)
    if top == "LATIN":
        return "latin"
    return {"CYRILLIC": "cyrillic", "HAN": "han", "HIRAGANA": "kana", "KATAKANA": "kana",
            "HANGUL": "hangul", "ARABIC": "arabic", "DEVANAGARI": "devanagari", "HEBREW": "hebrew"}[top]


def _script(text: str) -> str:
    counts = {}
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            for key in ("CYRILLIC", "HAN", "HIRAGANA", "KATAKANA", "HANGUL", "ARABIC", "DEVANAGARI", "HEBREW", "LATIN"):
                if key in name:
                    counts[key] = counts.get(key, 0) + 1
                    break
    if not counts:
        return "latin"
    top = max(counts, key=counts.get)
    if top == "LATIN":
        return "latin"
    return {"CYRILLIC": "cyrillic", "HAN": "han", "HIRAGANA": "kana", "KATAKANA": "kana",
            "HANGUL": "hangul", "ARABIC": "arabic", "DEVANAGARI": "devanagari", "HEBREW": "hebrew"}[top]


def detect_lang(text: str) -> str:
    """Heuristic ISO-639-1 guess; returns 'default' (global threshold) when unsure."""
    s = _script(text)
    if s == "cyrillic":
        uk_score = sum(1 for w in re.findall(r"[a-zа-яіїєґ']+", text.lower()) if w in STOPWORDS["uk"])
        uk_score += 2 * sum(text.count(c) for c in "ґєії")
        ru_score = sum(1 for w in re.findall(r"[a-zа-яёъыэ']+", text.lower()) if w in STOPWORDS["ru"])
        return "uk" if uk_score >= MIN_SCORE and uk_score > ru_score else ("ru" if ru_score >= MIN_SCORE else "default")
    if s == "han":
        return "zh"
    if s == "kana":
        return "ja"
    if s == "hangul":
        return "ko"
    if s == "arabic":
        return "ar"
    if s == "devanagari":
        return "hi"
    if s == "hebrew":
        return "he"
    words = re.findall(r"[a-zà-ÿěščřžďťňáéíúůóßäöüñãõ]+", text.lower())
    if not words:
        return "default"
    scores = {}
    for lang, stops in STOPWORDS.items():
        if lang in ("ru", "uk"):
            continue
        n = sum(1 for w in words if w in stops)
        n += 2 * sum(text.lower().count(c) for c in DIACRITICS.get(lang, ""))
        scores[lang] = n
    best = max(scores, key=scores.get)
    return best if scores[best] >= MIN_SCORE else "default"



def load_state_dict_any(directory) -> dict:
    """Load model weights from model_int8.safetensors (dequantized) if present,
    else model_fp16.safetensors, else model.safetensors."""
    from pathlib import Path as _P
    from safetensors.torch import load_file

    d = _P(directory)
    if (d / "model_int8.safetensors").exists():
        st = load_file(str(d / "model_int8.safetensors"))
        out = {}
        for k, v in st.items():
            if k.endswith(".q"):
                base = k[:-2]
                out[base] = (v.float() * st[base + ".scale"]).to(torch.float32)
            elif not k.endswith(".scale"):
                out[k] = v
        return out
    if (d / "model_fp16.safetensors").exists():
        return {k: v.float() for k, v in load_file(str(d / "model_fp16.safetensors")).items()}
    return load_file(str(d / "model.safetensors"))


def remap_hf_state(state: dict) -> dict:
    """Standard Llama-layout safetensors -> training-layout state dict."""
    import re

    out = {"embed.weight": state["model.embed_tokens.weight"]}
    layer_ids = sorted({int(m.group(1)) for k in state if (m := re.match(r"model\.layers\.(\d+)\.", k))})
    for i in layer_ids:
        p, b = f"model.layers.{i}.", f"blocks.{i}."
        qkv = torch.cat(
            [state[p + "self_attn.q_proj.weight"], state[p + "self_attn.k_proj.weight"], state[p + "self_attn.v_proj.weight"]],
            dim=0,
        )
        out[b + "attn.qkv.weight"] = qkv.contiguous()
        out[b + "attn.proj.weight"] = state[p + "self_attn.o_proj.weight"]
        out[b + "mlp.gate.weight"] = state[p + "mlp.gate_proj.weight"]
        out[b + "mlp.up.weight"] = state[p + "mlp.up_proj.weight"]
        out[b + "mlp.down.weight"] = state[p + "mlp.down_proj.weight"]
        out[b + "norm1.weight"] = state[p + "input_layernorm.weight"]
        out[b + "norm2.weight"] = state[p + "post_attention_layernorm.weight"]
    out["norm.weight"] = state["model.norm.weight"]
    if "score.weight" in state:
        out["score.weight"] = state["score.weight"]
    return out


def internal_cfg(cfg: dict) -> dict:
    """config.json (Llama layout) -> the dict the Censer module expects."""
    return {
        "vocab_size": cfg["vocab_size"],
        "n_layers": cfg["num_hidden_layers"],
        "n_heads": cfg["num_attention_heads"],
        "d_model": cfg["hidden_size"],
        "d_ff": cfg["intermediate_size"],
        "max_seq_len": cfg["max_position_embeddings"],
        "dropout": cfg.get("dropout", 0.0),
        "tie_embeddings": True,
        "pad_id": cfg.get("pad_token_id", 0),
        "answer_ids": cfg["answer_ids"],
        "max_chars": cfg.get("max_chars", 4000),
        "rope_scaling": cfg.get("rope_scaling") or (cfg.get("rope_parameters") if cfg.get("rope_parameters", {}).get("rope_type") == "linear" else None),
        "categories": [cfg["id2label"][str(i)] for i in range(2, cfg.get("num_labels", 2))] if cfg.get("num_labels", 2) > 2 else [],
        "cat_thresholds": cfg.get("cat_thresholds") or {},
        "temperature": float(cfg.get("temperature", 1.0) or 1.0),
        "lang_thresholds": cfg.get("lang_thresholds") or {},
    }



class CenserPipeline:
    """Text in, "1" (unsafe) or "0" (safe) out."""

    def __init__(self, directory: str | Path = ".", threshold: float = 0.5, device: str | None = None, prefer_quant: bool = False) -> None:
        directory = Path(directory)
        self.cfg = internal_cfg(json.loads((directory / "config.json").read_text(encoding="utf-8")))
        self.tok = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.pad_id = self.tok.token_to_id(PAD)
        self.bos_id = self.tok.token_to_id(BOS)
        self.user_id = self.tok.token_to_id(USER)
        self.assistant_id = self.tok.token_to_id(ASSISTANT)
        self.threshold = threshold
        self.cat_thresholds = dict(self.cfg.get("cat_thresholds") or {})
        self.temperature = float(self.cfg.get("temperature", 1.0) or 1.0)
        self.cat_experimental = set(self.cfg.get("cat_experimental") or [])
        # Barist.7: category-head hard block disabled until the head passes its own
        # gate (category recall >= 0.8 on heldout2 for sexual_abuse/self_harm).
        self.category_block = bool(self.cfg.get("category_block", False))
        self.lang_thresholds = dict(self.cfg.get("lang_thresholds") or {})
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = Censer(self.cfg)
        if prefer_quant and (directory / "model_int8.safetensors").exists():
            state = remap_hf_state(load_state_dict_any(directory))
        else:
            from safetensors.torch import load_file
            state = remap_hf_state(load_file(directory / "model.safetensors"))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        missing = [k for k in missing if k != "lm_head.weight"]  # tied to embed.weight
        if missing or unexpected:
            raise RuntimeError(f"weight mismatch: missing={missing} unexpected={unexpected}")
        self.model.to(self.device).eval()
        self.max_chars = self.cfg["max_chars"]

    def _encode(self, text: str) -> list[int]:
        body = self.tok.encode(text[: self.max_chars], add_special_tokens=False).ids
        ids = [self.bos_id, self.user_id, *body, self.assistant_id]
        return ids[: self.cfg["max_seq_len"]]

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def _encode_full(self, text: str) -> list[int]:
        body = self.tok.encode(text[: self.max_chars * 8], add_special_tokens=False).ids
        return [self.bos_id, self.user_id, *body, self.assistant_id]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        # long inputs: split into overlapping token chunks, aggregate by max
        out: list[dict] = []
        chunk_tokens, stride = 256, 192
        for text in texts:
            thr = self.threshold_for(text)
            ids = self._encode_full(text)
            body = ids[2:-1]
            if len(body) <= chunk_tokens:
                out.append(self._classify_ids([ids[: self.cfg["max_seq_len"]]], [thr], [text])[0])
                continue
            chunks = []
            start = 0
            while start < len(body):
                chunks.append(body[start:start + chunk_tokens])
                if start + chunk_tokens >= len(body):
                    break
                start += stride
            rows = []
            for piece in chunks:
                cids = [self.bos_id, self.user_id, *piece, self.assistant_id]
                rows.append(self._classify_ids([cids[: self.cfg["max_seq_len"]]], [thr], [text])[0])
            best = max(rows, key=lambda r: r["score"])
            # aggregate categories by max prob
            n_cat = len(rows[0].get("category_scores", []))
            cat_scores = [max(r["category_scores"][i] for r in rows) for i in range(n_cat)] if n_cat else []
            best["category_scores"] = [round(c, 4) for c in cat_scores]
            cats = self.categories()
            if best["unsafe"] and cats:
                passed = {c for c, p in zip(cats, cat_scores)
                          if p >= float(self.cat_thresholds.get(c, self.threshold)) and c not in self.cat_experimental}
                passed |= set(best.get("categories") or [])  # keep pre-filter hits from the best chunk
                # hard block (item 1): sexual_abuse / self_harm >= 0.3 forces unsafe —
                # Barist.7: only when category_block is enabled in config (head gate)
                if self.category_block and not best.get("needs_support"):
                    for c in ("sexual_abuse", "self_harm"):
                        if c in cats and cat_scores[cats.index(c)] >= 0.3:
                            passed.add(c)
                            best["unsafe"] = True
                            best["label"] = "unsafe"
                            best["label_id"] = "1"
                            best["override"] = best.get("override") or "category_block"
                passed = sorted(passed, key=lambda c: (_cat_rank(c) if c in CAT_PRIORITY else 99,
                                                       -(cat_scores[cats.index(c)] if c in cats else 1.0)))[:2]
                if not passed:  # unsafe must always carry >=1 category: top-1 by score
                    passed = [max(cats, key=lambda c: cat_scores[cats.index(c)])]
                best["categories"] = passed
            else:
                best["categories"] = []
            best["chunks"] = len(chunks)
            out.append(best)
        return out

    def threshold_for(self, text: str) -> float:
        lang = detect_lang(text)
        return float(self.lang_thresholds.get(lang) or self.lang_thresholds.get("default") or self.threshold)

    def categories(self) -> list[str]:
        return self.cfg.get("categories") or []

    def _classify_ids(self, seqs: list[list[int]], thresholds: list[float] | None = None, texts: list[str] | None = None) -> list[dict]:
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        for i, seq in enumerate(seqs):
            ids[i, : len(seq)] = torch.tensor(seq)
        probs = self.model.classify_probs(ids.to(self.device), temperature=self.temperature).float().cpu().tolist()
        if thresholds is None:
            thresholds = [self.threshold] * len(seqs)
        res = []
        for i, row in enumerate(probs):
            p = row[1]
            cats = self.categories()
            cat_scores = row[2:] if cats else []
            is_unsafe = p >= thresholds[i]
            # Barist.7: must-catch pre-filter — CSAM / self-harm pattern hits force
            # unsafe regardless of the (language) binary threshold
            hit_override = must_catch_hit(texts[i]) if texts else []
            review = review_hit(texts[i]) if texts else []
            help_ctx = is_support_context(texts[i]) if texts else False
            support_ctx = help_ctx
            # Barist.7: model-based self-harm >= 0.3 -> needs_support (NOT a block)
            if cats and "self_harm" in cats and cat_scores[cats.index("self_harm")] >= 0.3:
                support_ctx = True
            # Barist.7: neutral/positive protected-group mention -> waive model FP (regex hits still block)
            id_pos = bool(texts and identity_positive(texts[i]))
            if self.category_block and cats and "self_harm" in cats and cat_scores[cats.index("self_harm")] >= 0.2:
                support_ctx = True  # model-based support routing only when head is trusted
            # hard block: sexual_abuse / self_harm >= 0.3 forces unsafe in ANY language
            # Barist.7: gated behind config flag category_block (head recall gate)
            cat_block = []
            if self.category_block and cats and not help_ctx:  # help-other frames never hard-blocked
                for c in ("sexual_abuse", "self_harm"):
                    if c in cats and cat_scores[cats.index(c)] >= 0.3:
                        cat_block.append(c)
            if cat_block:
                is_unsafe = True
            if hit_override:
                is_unsafe = True
            if help_ctx and not hit_override and not cat_block:
                is_unsafe = False  # help-seeking context is never blocked
            if id_pos and not hit_override:
                is_unsafe = False  # identity-positive waiver
            # Barist.7: at most the top-2 categories by score (no nonsense lists)
            _passed = [c for c, sc in zip(cats, cat_scores)
                       if sc >= float(self.cat_thresholds.get(c, self.threshold))
                       and c not in self.cat_experimental]
            _passed = sorted(set(_passed) | set(hit_override) | set(cat_block),
                             key=lambda c: (_cat_rank(c) if c in CAT_PRIORITY else 99,
                                            -(cat_scores[cats.index(c)] if c in cats else 1.0)))[:2]
            if is_unsafe and not _passed:
                # unsafe must always carry >=1 category: top-1 by score
                _passed = [max(cats, key=lambda c: cat_scores[cats.index(c)])] if cats else []
            res.append({
                "label": "unsafe" if is_unsafe else "safe",
                "label_id": "1" if is_unsafe else "0",
                "unsafe": is_unsafe,
                "override": "must_catch" if hit_override else ("category_block" if cat_block else None),
                "needs_support": SUPPORT_LINK if (support_ctx or "self_harm" in hit_override) else None,
                "needs_review": True if review else None,
                "score": round(p, 4),
                "category_scores": [round(c, 4) for c in cat_scores],
                # categories are only reported when the binary verdict is unsafe
                "categories": _passed if is_unsafe else [],
            })
        return res




def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def repl(pipe: "CenserPipeline") -> int:
    """Small interactive terminal UI: phrase in, safe/unsafe + score out."""
    print("Barist.7 - interactive terminal UI")
    print("Type a phrase, Enter to classify. 'exit' to quit.\n")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        text = line.strip()
        if text.lower() in {"exit", "quit", ":q"}:
            return 0
        if not text:
            continue
        v = pipe.predict(text)
        label = "unsafe" if v["unsafe"] else "safe"
        colored = _color(f"{label:6s}", "31" if v["unsafe"] else "32")
        print(f"{colored} {v['score']:.4f} | {text}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="barist.7",
        description="Barist.7 safety classifier (standalone). Run without text to open the interactive UI.",
    )
    parser.add_argument("text", nargs="*", help="phrase to classify once; omit to open the interactive UI")
    parser.add_argument("--model-dir", default=".")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--prefer-quant", action="store_true",
                        help="load model_int8.safetensors (0.9 MB) instead of fp32 if present")
    parser.add_argument("--repl", action="store_true", help="interactive UI (default when no text is given)")
    args = parser.parse_args(argv)

    pipe = CenserPipeline(args.model_dir, args.threshold, prefer_quant=args.prefer_quant)
    text = " ".join(args.text).strip()
    if args.repl or not text:
        raise SystemExit(repl(pipe))
    verdict = pipe.predict(text)
    print(json.dumps(verdict) if args.json else verdict["label"])  # "safe" / "unsafe"


if __name__ == "__main__":
    main()

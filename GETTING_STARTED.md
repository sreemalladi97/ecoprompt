# EcoPrompt — Getting Started for the Team

EcoPrompt is our middleware proxy that cuts AI costs by routing prompts to the
right-sized free model, caching repeat questions, and compressing prompts
before they're sent. This doc gets you set up to explore what's live.

## Live links

- **Dashboard** (usage stats, routing breakdown): https://ecoprompt-nu.vercel.app/dashboard
- **Tester** (send prompts, see how they get routed): https://ecoprompt-nu.vercel.app/test

Both work from any browser, no setup required on your end beyond a free API key (next step).

## 1. Get a free Groq API key

We route to free models hosted on Groq. You'll need your own key to test:

1. Go to [console.groq.com](https://console.groq.com/login) and sign up (email, Google, or GitHub — no credit card needed)
2. Go to [console.groq.com/keys](https://console.groq.com/keys)
3. Click **Create API Key**, name it whatever you like, and **copy it immediately** — Groq only shows it once

## 2. Try the tester

Open the [tester link](https://ecoprompt-nu.vercel.app/test) and:

1. Paste your Groq key into the **Groq API Key** field (it's remembered in your browser after that, you won't need to re-paste it each visit)
2. Click one of the three example buttons — **simple**, **medium**, or **complex** — or type your own prompt
3. Click **Send request**

The result panel shows:
- **Tier (why)** — which bucket (simple/medium/complex) the prompt got routed to, and which specific rule decided it (word count or a matched keyword)
- **Model used** — the actual model that answered
- **Cache** — whether it was a fresh answer or served from cache
- **Tokens in/out** and **tokens saved** — usage for that request

Try a few of your own prompts and see if the tier it picks feels right. If something looks mis-routed, the "why" line tells you exactly which rule fired, that's useful feedback for tuning the routing logic later.

## 3. Check the dashboard

The [dashboard](https://ecoprompt-nu.vercel.app/dashboard) shows aggregate stats: total requests, cache hit rate, and how many requests went to each tier. Numbers reset periodically on this deployment (see note below), so don't worry if they look low.

## Known limitations on this deployment

This is running on Vercel, which is great for a shareable link but has some trade-offs versus running EcoPrompt locally:

- **Semantic caching and prompt compression are inactive.** Those features need persistent memory that Vercel's serverless model doesn't provide, so those specific stats won't reflect real savings here. Routing still works fully.
- **Stats reset periodically.** Vercel spins the app down when idle, which clears in-memory counters. Don't read too much into long-term trends from the dashboard yet.

## Want to run it locally or contribute?

The full project (including the working semantic cache and compression) lives at
**github.com/sreemalladi97/ecoprompt** (private repo, ask Sree for access if you don't have it).

Quick local setup:
```bash
git clone https://github.com/sreemalladi97/ecoprompt.git
cd ecoprompt
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-local.txt
echo "GROQ_API_KEY=your_key_here" > .env   # use your own key from console.groq.com/keys
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then visit `http://localhost:8000/dashboard` and `http://localhost:8000/test` locally, with caching and compression active this time (unlike the Vercel deployment above, which intentionally skips `requirements-local.txt` to stay fast on serverless cold starts).

Questions? Ping Sree.

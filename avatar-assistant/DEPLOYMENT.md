# Deployment Guide

Fast, free-tier path: **Render** for the backend, **Vercel** for the
frontend. Both have generous free tiers and deploy straight from a Git
repo — no credit card needed for this scale of project. Budget ~30-40
minutes including the RAG index rebuild.

## 0. Push to GitHub first
Both Render and Vercel deploy from a Git repo, not a zip upload.
```bash
cd avatar-assistant
git init
git add .
git commit -m "Personal Avatar Assistant MVP"
```
Create a new repo on GitHub, then:
```bash
git remote add origin https://github.com/<you>/<repo-name>.git
git branch -M main
git push -u origin main
```

## 1. Backend on Render
1. Go to https://render.com → **New +** → **Web Service** → connect your
   GitHub repo.
2. **Root directory:** `backend`
3. **Build command:**
   ```
   pip install -r requirements.txt && python scripts/ingest.py
   ```
   (Running `ingest.py` as part of the build means the RAG index gets
   rebuilt fresh on every deploy — necessary because Render's free tier
   has an ephemeral filesystem, so anything not rebuilt at deploy time
   won't survive a restart.)
4. **Start command:**
   ```
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
5. **Environment variables** (Render dashboard → Environment):
   - `OPENAI_API_KEY` = your real key
   - `APP_ENV` = `production`
   - `FRONTEND_URL` = *(fill in after step 2, then redeploy)*
6. Deploy. First build takes a few minutes (downloading the
   `sentence-transformers` model + PyTorch). Once live, note the URL —
   something like `https://your-app.onrender.com`.
7. Sanity check: visit `https://your-app.onrender.com/health` — you
   should see the same JSON as your local `/health` check.

**Free tier note:** Render's free web services spin down after 15
minutes of inactivity and take ~30-60 seconds to wake back up on the
next request. For a live demo, hit the `/health` URL a minute before you
present to "wake it up" and avoid an awkward pause.

## 2. Frontend on Vercel
1. Go to https://vercel.com → **Add New** → **Project** → import the same
   GitHub repo.
2. **Root directory:** `frontend`
3. **Framework preset:** Vite (should auto-detect)
4. **Environment variable:**
   - `VITE_BACKEND_URL` = your Render URL from step 1 (e.g.
     `https://your-app.onrender.com`)
5. Deploy. Vercel gives you a URL like `https://your-app.vercel.app`.

## 3. Close the loop on CORS
Go back to Render → Environment → set `FRONTEND_URL` to your Vercel URL
from step 2 → save (this triggers a redeploy automatically). Without
this, the browser will block requests from your deployed frontend to
your deployed backend.

## 4. Final verification
Open your Vercel URL in an incognito window (to rule out any local
caching) and walk through the acceptance checklist:
- [ ] Avatar loads and idles
- [ ] Chat: ask a grounded question, get a real answer
- [ ] Chat: ask an unrelated question, get an honest "I don't know"
- [ ] Voice: hold mic, ask a question, hear the spoken answer
- [ ] Camera: capture an object, get a correct description, spoken aloud
- [ ] Text input still works if you deny mic/camera permissions

## Alternative: skip hosting, demo locally
If deployment eats into time you don't have, running it locally
(`npm run dev` + `uvicorn --reload`) on the presentation laptop is a
completely legitimate fallback — nothing in the plan requires public
hosting to work during the actual live demo, only that a public link
*exists* for the submission. If you go this route, still do step 0
(push to GitHub) so the repository link requirement is met, and lean more
on the backup demo video (see `DEMO_CHECKLIST.md`) as your safety net in
case the projector setup has network issues.

## Common deployment issues
- **Backend deploys but `/chat` returns 503** → the build command's
  `ingest.py` step failed silently; check Render's build logs for a
  download error partway through `sentence-transformers` setup.
- **CORS error on the deployed frontend, works fine locally** → you
  skipped step 3, or forgot to redeploy the backend after setting
  `FRONTEND_URL`.
- **Frontend shows "Backend unreachable" right after deploy** → Render
  free tier is asleep; wait ~60 seconds and refresh, or hit `/health`
  directly first.
- **Avatar doesn't load on the deployed site but works locally** → check
  that `public/models/RobotExpressive.glb` was actually committed to
  Git — large binary files are sometimes accidentally gitignored; run
  `git ls-files | grep glb` to confirm it's tracked.

# Complete Deployment Guide: Clinical Document ML System

This guide outlines **4 deployment methods** ranging from instant public tunnels (for viva presentation) to containerized cloud deployments.

---

## Option 1: Instant Live Public URL (Zero Setup, Free)
*Ideal for showing your professor right now from your phone or their laptop.*

You can expose your local port `8000` to the internet instantly using a secure tunnel:

### Method A: Via Cloudflare Tunnel (No account needed)
1. Download `cloudflared.exe` from Cloudflare or run:
   ```bash
   winget install --id Cloudflare.cloudflared
   ```
2. Start the tunnel to your running port:
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
3. Copy the generated public link (e.g. `https://random-subdomain.trycloudflare.com`) and open it on any device.

### Method B: Via Ngrok
1. Run:
   ```bash
   ngrok http 8000
   ```
2. Open the resulting `https://your-id.ngrok-free.app` URL.

---

## Option 2: Deploy Free on Hugging Face Spaces (Best for ML Viva)
Hugging Face Spaces is recognized by academic evaluators as the standard repository for ML apps.

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Click **New Space** $\rightarrow$ select **Docker** as the Space SDK (Blank).
3. Set Space visibility to **Public**.
4. Push your project code to the Space's Git repository:
   ```bash
   git init
   git remote add origin https://huggingface.co/spaces/<your-username>/<your-space-name>
   git add .
   git commit -m "Deploy Clinical STG Pipeline"
   git push origin main
   ```
5. Hugging Face will automatically detect the [`Dockerfile`](Dockerfile), build the container, install Tesseract OCR and dependencies, and launch your app with a permanent HTTPS link.

---

## Option 3: Deploy Free on Render.com
1. Push this folder to a GitHub repository.
2. Sign in to [Render.com](https://render.com) $\rightarrow$ Click **New +** $\rightarrow$ **Web Service**.
3. Connect your GitHub repository.
4. Select environment: **Docker**.
5. Click **Create Web Service**. Render builds the `Dockerfile` and gives you a free live URL: `https://your-app.onrender.com`.

---

## Option 4: Local / Server Docker Deployment (Production)

To run the application inside an isolated Docker container on any machine (Windows, Linux, Mac):

1. **Build the Docker Image:**
   ```bash
   docker build -t clinical-stg-app .
   ```
2. **Run the Container:**
   ```bash
   docker run -d -p 8000:8000 --name clinical_ai clinical-stg-app
   ```
3. **Or run with Docker Compose:**
   ```bash
   docker compose up -d
   ```
4. Access the application at `http://localhost:8000`.

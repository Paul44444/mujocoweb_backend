# Hosting the backend on this Ubuntu computer

Render support remains unchanged. The same `server.py`, `Dockerfile`, and
environment variables work on Render and locally.

## 1. Local configuration

Create the untracked `.env` file from the template and insert the OpenAI key:

```bash
cp .env.example .env
```

Start the backend on the loopback interface:

```bash
./run-local.sh
```

Verify it locally at `http://127.0.0.1:8000`. Binding only to loopback is
intentional: the tunnel can reach it, while the router cannot expose it
directly.

## 2. Persistent startup

The file `deploy/mujocoweb-backend.service` is a ready-to-use systemd unit for
this computer. Install and enable it with:

```bash
sudo cp deploy/mujocoweb-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mujocoweb-backend
sudo systemctl status mujocoweb-backend
```

Follow logs with:

```bash
journalctl -u mujocoweb-backend -f
```

## 3. Public HTTPS/WSS address

Use a named Cloudflare Tunnel for a stable public hostname. In Cloudflare,
create a tunnel and route its public hostname to:

```text
http://localhost:8000
```

Install the generated tunnel command as a system service on this computer.
No inbound router port should be opened.

For a disposable test, Cloudflare's quick-tunnel command is:

```bash
cloudflared tunnel --url http://localhost:8000
```

Quick-tunnel hostnames change after restarting and are not intended for
production.

## 4. Switch the frontend

Set `VITE_BACKEND_URL` in Vercel to the tunnel's `https://...` URL and redeploy.
Remove that variable and redeploy to return to Render. The frontend repository's
`HOSTING.md` also documents a browser-only override for temporary tests.

# Sound-to-3D Server

Backend server for synchronizing training data across browsers and devices.

## Deployment to Render

### Step 1: Push to GitHub
```bash
cd /Users/ihwain/sound-to-3d
git add server/
git commit -m "Add backend server for data synchronization"
git push
```

### Step 2: Deploy on Render

1. Go to [https://render.com](https://render.com) and sign in
2. Click "New +" → "Web Service"
3. Connect your GitHub repository: `audsilhouette25/sound-to-3d`
4. Configure the service:
   - **Name**: `sound-to-3d-server`
   - **Region**: Choose closest to you
   - **Branch**: `0110수정(지원)` (or your main branch)
   - **Root Directory**: `server`
   - **Runtime**: Node
   - **Build Command**: `npm install`
   - **Start Command**: `npm start`
   - **Plan**: Free

5. Click "Create Web Service"

6. Wait for deployment (5-10 minutes)

7. Copy your service URL (e.g., `https://sound-to-3d-server.onrender.com`)

### Step 3: Update Frontend

1. Open `sketch.js`
2. Update line 17-18:
```javascript
const API_URL = 'https://sound-to-3d-server.onrender.com'; // Your Render URL
const USE_SERVER = true; // Enable server
```

3. Commit and push:
```bash
git add sketch.js
git commit -m "Enable server synchronization"
git push
```

## Testing

After deployment:
1. Open the app in Chrome
2. Add some training data
3. Open the app in Safari or another browser
4. You should see the same data!

## API Endpoints

- `GET /api/data` - Get all training data
- `POST /api/data` - Add new training data
- `DELETE /api/data` - Clear all training data
- `GET /health` - Health check

## Notes

- Free tier on Render may sleep after 15 minutes of inactivity
- First request after sleep takes ~30 seconds to wake up
- Data persists even when service sleeps
- For production, consider upgrading to paid tier

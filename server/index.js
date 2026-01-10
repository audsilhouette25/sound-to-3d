const express = require('express');
const cors = require('cors');
const fs = require('fs').promises;
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const DATA_FILE = path.join(__dirname, 'training-data.json');

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Initialize data file if it doesn't exist
async function initDataFile() {
    try {
        await fs.access(DATA_FILE);
    } catch {
        await fs.writeFile(DATA_FILE, JSON.stringify({ data: [] }));
    }
}

// GET /api/data - Get all training data
app.get('/api/data', async (req, res) => {
    try {
        const fileContent = await fs.readFile(DATA_FILE, 'utf-8');
        const { data } = JSON.parse(fileContent);
        res.json({ success: true, data });
    } catch (error) {
        console.error('Error reading data:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

// POST /api/data - Add new training data
app.post('/api/data', async (req, res) => {
    try {
        const newData = req.body;

        // Read existing data
        const fileContent = await fs.readFile(DATA_FILE, 'utf-8');
        const { data } = JSON.parse(fileContent);

        // Add new data
        data.push(newData);

        // Write back to file
        await fs.writeFile(DATA_FILE, JSON.stringify({ data }, null, 2));

        res.json({ success: true, count: data.length });
    } catch (error) {
        console.error('Error adding data:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

// DELETE /api/data - Clear all training data
app.delete('/api/data', async (req, res) => {
    try {
        await fs.writeFile(DATA_FILE, JSON.stringify({ data: [] }, null, 2));
        res.json({ success: true, message: 'All data cleared' });
    } catch (error) {
        console.error('Error clearing data:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

// Health check
app.get('/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Start server
async function start() {
    await initDataFile();
    app.listen(PORT, () => {
        console.log(`🚀 Server running on port ${PORT}`);
        console.log(`📊 Data file: ${DATA_FILE}`);
    });
}

start().catch(console.error);

const express = require('express')
const app = express()
const path = require('path');

// Serve static files
app.use('/static', express.static(path.join(__dirname, 'public')));
app.listen(3000)
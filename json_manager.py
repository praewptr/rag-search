#!/usr/bin/env python3
"""
Standalone JSON Manager for RAG PDF System
Run this file to start a web server with a UI for managing your PDF JSON data.
"""

import json
import os
import webbrowser
from threading import Timer
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import mimetypes

# Set the port for the web server
PORT = 8080
JSON_FILE_PATH = "pdf_map.json"

class JSONManagerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == "/" or parsed_path.path == "/index.html":
            self.serve_html()
        elif parsed_path.path == "/api/load":
            self.load_json_data()
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == "/api/save":
            self.save_json_data()
        elif parsed_path.path == "/api/add":
            self.add_json_entry()
        elif parsed_path.path == "/api/delete":
            self.delete_json_entry()
        else:
            self.send_error(404)
    
    def serve_html(self):
        """Serve the main HTML page"""
        html_content = self.get_html_content()
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))
    
    def load_json_data(self):
        """Load and return JSON data"""
        try:
            if os.path.exists(JSON_FILE_PATH):
                with open(JSON_FILE_PATH, 'r', encoding='utf-8') as file:
                    data = json.load(file)
            else:
                data = []
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        except Exception as e:
            self.send_error(500, f"Error loading JSON: {str(e)}")
    
    def save_json_data(self):
        """Save JSON data to file"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            with open(JSON_FILE_PATH, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
        except Exception as e:
            self.send_error(500, f"Error saving JSON: {str(e)}")
    
    def add_json_entry(self):
        """Add a new entry to JSON data"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            new_entry = json.loads(post_data.decode('utf-8'))
            
            # Load existing data
            if os.path.exists(JSON_FILE_PATH):
                with open(JSON_FILE_PATH, 'r', encoding='utf-8') as file:
                    data = json.load(file)
            else:
                data = []
            
            # Add new entry
            data.append(new_entry)
            
            # Save updated data
            with open(JSON_FILE_PATH, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "data": data}).encode('utf-8'))
        except Exception as e:
            self.send_error(500, f"Error adding entry: {str(e)}")
    
    def delete_json_entry(self):
        """Delete an entry from JSON data"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            entry_id = request_data.get('id')
            
            # Load existing data
            if os.path.exists(JSON_FILE_PATH):
                with open(JSON_FILE_PATH, 'r', encoding='utf-8') as file:
                    data = json.load(file)
            else:
                data = []
            
            # Remove entry with matching id
            data = [item for item in data if item.get('id') != entry_id]
            
            # Save updated data
            with open(JSON_FILE_PATH, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "data": data}).encode('utf-8'))
        except Exception as e:
            self.send_error(500, f"Error deleting entry: {str(e)}")
    
    def get_html_content(self):
        """Return the complete HTML page with embedded CSS and JavaScript"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JSON Manager - RAG PDF System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 16px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            padding: 30px;
        }
        
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .section {
            background: white;
            margin-bottom: 30px;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
            border: 1px solid #e2e8f0;
        }
        
        .section h2 {
            color: #34495e;
            margin-bottom: 20px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
            font-size: 1.5rem;
        }
        
        .form-row {
            display: grid;
            grid-template-columns: 1fr 2fr 1fr;
            gap: 15px;
            margin-bottom: 15px;
            align-items: end;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
        }
        
        .form-group label {
            margin-bottom: 5px;
            font-weight: 600;
            color: #555;
        }
        
        .form-group input {
            padding: 12px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s, box-shadow 0.3s;
        }
        
        .form-group input:focus {
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
        }
        
        .btn-primary:hover {
            background: linear-gradient(135deg, #2980b9, #3498db);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
        }
        
        .btn-success {
            background: linear-gradient(135deg, #27ae60, #16a085);
            color: white;
        }
        
        .btn-success:hover {
            background: linear-gradient(135deg, #16a085, #27ae60);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
        }
        
        .btn-danger {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            color: white;
            padding: 8px 16px;
            font-size: 12px;
        }
        
        .btn-danger:hover {
            background: linear-gradient(135deg, #c0392b, #e74c3c);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(231, 76, 60, 0.3);
        }
        
        .btn-outline {
            background: transparent;
            border: 2px solid #3498db;
            color: #3498db;
        }
        
        .btn-outline:hover {
            background: #3498db;
            color: white;
            transform: translateY(-2px);
        }
        
        .json-list {
            max-height: 400px;
            overflow-y: auto;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            padding: 15px;
            background: #f8f9fa;
        }
        
        .json-item {
            background: white;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .json-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        .json-item:last-child {
            margin-bottom: 0;
        }
        
        .json-item-info {
            flex: 1;
        }
        
        .json-item-info h4 {
            color: #2c3e50;
            margin-bottom: 8px;
            font-size: 1.1rem;
        }
        
        .json-item-info p {
            color: #666;
            margin-bottom: 4px;
        }
        
        .json-item-info a {
            color: #3498db;
            text-decoration: none;
            word-break: break-all;
        }
        
        .json-item-info a:hover {
            text-decoration: underline;
        }
        
        .json-editor {
            margin-top: 20px;
        }
        
        .json-editor textarea {
            width: 100%;
            min-height: 200px;
            padding: 15px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            background: #f8f9fa;
            resize: vertical;
        }
        
        .json-editor textarea:focus {
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }
        
        .json-actions {
            margin-top: 15px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .status-message {
            margin-top: 15px;
            padding: 15px;
            border-radius: 8px;
            display: none;
            font-weight: 500;
        }
        
        .status-success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status-error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
        }
        
        .stat-card h3 {
            font-size: 2rem;
            margin-bottom: 5px;
        }
        
        .stat-card p {
            opacity: 0.9;
        }
        
        @media (max-width: 768px) {
            .form-row {
                grid-template-columns: 1fr;
            }
            
            .json-actions {
                justify-content: center;
            }
            
            .container {
                padding: 15px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🗂️ JSON Manager - RAG PDF System</h1>
        
        <!-- Stats Section -->
        <div class="stats">
            <div class="stat-card">
                <h3 id="totalCount">0</h3>
                <p>Total Documents</p>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #27ae60, #16a085);">
                <h3 id="validUrlCount">0</h3>
                <p>Valid URLs</p>
            </div>
        </div>
        
        <!-- Add New Entry Section -->
        <div class="section">
            <h2>➕ Add New PDF Document</h2>
            <form id="addForm">
                <div class="form-row">
                    <div class="form-group">
                        <label for="newId">Document ID:</label>
                        <input type="text" id="newId" placeholder="e.g., pdf003" required>
                    </div>
                    <div class="form-group">
                        <label for="newTitle">Document Title:</label>
                        <input type="text" id="newTitle" placeholder="e.g., Manual Guide.pdf" required>
                    </div>
                    <div class="form-group">
                        <button type="submit" class="btn btn-primary">Add Document</button>
                    </div>
                </div>
                <div class="form-group">
                    <label for="newUrl">Document URL:</label>
                    <input type="url" id="newUrl" placeholder="https://example.com/document.pdf" required>
                </div>
            </form>
        </div>
        
        <!-- Document List Section -->
        <div class="section">
            <h2>📚 Current Documents</h2>
            <div id="jsonList" class="json-list">
                <!-- Documents will be loaded here -->
            </div>
            <button id="refreshBtn" class="btn btn-outline">🔄 Refresh List</button>
        </div>
        
        <!-- JSON Editor Section -->
        <div class="section">
            <h2>⚡ Advanced JSON Editor</h2>
            <div class="json-editor">
                <textarea id="jsonEditor" placeholder="JSON data will appear here..."></textarea>
                <div class="json-actions">
                    <button id="formatBtn" class="btn btn-outline">🎨 Format JSON</button>
                    <button id="validateBtn" class="btn btn-outline">✅ Validate JSON</button>
                    <button id="saveBtn" class="btn btn-success">💾 Save Changes</button>
                    <button id="downloadBtn" class="btn btn-outline">📥 Download JSON</button>
                </div>
                <div id="statusMessage" class="status-message"></div>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let currentData = [];
        
        // Initialize the application
        document.addEventListener('DOMContentLoaded', function() {
            loadData();
            initializeEventListeners();
        });
        
        function initializeEventListeners() {
            // Form submission
            document.getElementById('addForm').addEventListener('submit', addNewEntry);
            
            // Button clicks
            document.getElementById('refreshBtn').addEventListener('click', loadData);
            document.getElementById('formatBtn').addEventListener('click', formatJSON);
            document.getElementById('validateBtn').addEventListener('click', validateJSON);
            document.getElementById('saveBtn').addEventListener('click', saveJSON);
            document.getElementById('downloadBtn').addEventListener('click', downloadJSON);
        }
        
        async function loadData() {
            try {
                showLoading(true);
                const response = await fetch('/api/load');
                const data = await response.json();
                currentData = data;
                renderDocumentList(data);
                updateEditor(data);
                updateStats(data);
                showStatus('Data loaded successfully!', 'success');
            } catch (error) {
                showStatus(`Error loading data: ${error.message}`, 'error');
            } finally {
                showLoading(false);
            }
        }
        
        function renderDocumentList(data) {
            const listContainer = document.getElementById('jsonList');
            
            if (data.length === 0) {
                listContainer.innerHTML = '<p style="text-align: center; color: #666; font-style: italic;">No documents found. Add your first document above!</p>';
                return;
            }
            
            listContainer.innerHTML = data.map(item => `
                <div class="json-item">
                    <div class="json-item-info">
                        <h4>${item.title || 'Unknown Title'}</h4>
                        <p><strong>ID:</strong> ${item.id || 'No ID'}</p>
                        <p><strong>URL:</strong> <a href="${item.url || '#'}" target="_blank">${item.url || 'No URL'}</a></p>
                    </div>
                    <button class="btn btn-danger" onclick="deleteEntry('${item.id}')">🗑️ Delete</button>
                </div>
            `).join('');
        }
        
        function updateStats(data) {
            document.getElementById('totalCount').textContent = data.length;
            const validUrls = data.filter(item => item.url && item.url.startsWith('http')).length;
            document.getElementById('validUrlCount').textContent = validUrls;
        }
        
        function updateEditor(data) {
            document.getElementById('jsonEditor').value = JSON.stringify(data, null, 2);
        }
        
        async function addNewEntry(event) {
            event.preventDefault();
            
            const newEntry = {
                id: document.getElementById('newId').value.trim(),
                title: document.getElementById('newTitle').value.trim(),
                url: document.getElementById('newUrl').value.trim()
            };
            
            // Validate required fields
            if (!newEntry.id || !newEntry.title || !newEntry.url) {
                showStatus('Please fill in all fields!', 'error');
                return;
            }
            
            // Check for duplicate ID
            if (currentData.some(item => item.id === newEntry.id)) {
                showStatus('ID already exists! Please use a different ID.', 'error');
                return;
            }
            
            try {
                showLoading(true);
                const response = await fetch('/api/add', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(newEntry)
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    currentData = result.data;
                    renderDocumentList(currentData);
                    updateEditor(currentData);
                    updateStats(currentData);
                    document.getElementById('addForm').reset();
                    showStatus('Document added successfully!', 'success');
                } else {
                    showStatus('Error adding document!', 'error');
                }
            } catch (error) {
                showStatus(`Error adding document: ${error.message}`, 'error');
            } finally {
                showLoading(false);
            }
        }
        
        async function deleteEntry(id) {
            if (!confirm(`Are you sure you want to delete the document with ID: ${id}?`)) {
                return;
            }
            
            try {
                showLoading(true);
                const response = await fetch('/api/delete', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ id: id })
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    currentData = result.data;
                    renderDocumentList(currentData);
                    updateEditor(currentData);
                    updateStats(currentData);
                    showStatus('Document deleted successfully!', 'success');
                } else {
                    showStatus('Error deleting document!', 'error');
                }
            } catch (error) {
                showStatus(`Error deleting document: ${error.message}`, 'error');
            } finally {
                showLoading(false);
            }
        }
        
        function formatJSON() {
            const editor = document.getElementById('jsonEditor');
            try {
                const jsonData = JSON.parse(editor.value);
                editor.value = JSON.stringify(jsonData, null, 2);
                showStatus('JSON formatted successfully!', 'success');
            } catch (error) {
                showStatus(`Invalid JSON: ${error.message}`, 'error');
            }
        }
        
        function validateJSON() {
            const editor = document.getElementById('jsonEditor');
            try {
                JSON.parse(editor.value);
                showStatus('JSON is valid! ✅', 'success');
            } catch (error) {
                showStatus(`Invalid JSON: ${error.message}`, 'error');
            }
        }
        
        async function saveJSON() {
            const editor = document.getElementById('jsonEditor');
            try {
                const jsonData = JSON.parse(editor.value);
                
                showLoading(true);
                const response = await fetch('/api/save', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(jsonData)
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    currentData = jsonData;
                    renderDocumentList(currentData);
                    updateStats(currentData);
                    showStatus('JSON saved successfully!', 'success');
                } else {
                    showStatus('Error saving JSON!', 'error');
                }
            } catch (error) {
                showStatus(`Error saving JSON: ${error.message}`, 'error');
            } finally {
                showLoading(false);
            }
        }
        
        function downloadJSON() {
            const editor = document.getElementById('jsonEditor');
            try {
                const jsonData = JSON.parse(editor.value);
                const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'pdf_map.json';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                showStatus('JSON downloaded successfully!', 'success');
            } catch (error) {
                showStatus(`Invalid JSON: ${error.message}`, 'error');
            }
        }
        
        function showStatus(message, type) {
            const statusEl = document.getElementById('statusMessage');
            statusEl.textContent = message;
            statusEl.className = `status-message status-${type}`;
            statusEl.style.display = 'block';
            
            setTimeout(() => {
                statusEl.style.display = 'none';
            }, 5000);
        }
        
        function showLoading(show) {
            const buttons = document.querySelectorAll('.btn');
            buttons.forEach(btn => {
                if (show) {
                    btn.disabled = true;
                    if (!btn.querySelector('.loading')) {
                        btn.innerHTML = '<span class="loading"></span>' + btn.innerHTML;
                    }
                } else {
                    btn.disabled = false;
                    const loading = btn.querySelector('.loading');
                    if (loading) {
                        loading.remove();
                    }
                }
            });
        }
        
        // Make deleteEntry globally accessible
        window.deleteEntry = deleteEntry;
    </script>
</body>
</html>'''

def open_browser():
    """Open the default web browser after a short delay"""
    webbrowser.open(f'http://localhost:{PORT}')

def run_server():
    """Start the HTTP server"""
    try:
        server = HTTPServer(('localhost', PORT), JSONManagerHandler)
        print(f"""
🚀 JSON Manager Server Started!

📱 Open your browser and go to: http://localhost:{PORT}
📄 JSON file: {os.path.abspath(JSON_FILE_PATH)}

Features:
✨ Add new PDF documents with ease
🗂️ View and manage existing documents
✏️ Edit JSON directly with syntax highlighting
💾 Auto-save your changes
📥 Download your JSON data
🔄 Real-time updates and validation

Press Ctrl+C to stop the server
        """)
        
        # Open browser after 1 second
        Timer(1.0, open_browser).start()
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Have a great day!")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    run_server()
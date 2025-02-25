document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('file-upload').addEventListener('change', function() {
        var fileName = this.files[0].name;
        document.getElementById('file-name').textContent = 'Selected file: ' + fileName;
    });
});

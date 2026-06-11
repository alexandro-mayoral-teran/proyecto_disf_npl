import os

css_path = os.path.join(os.path.dirname(__file__), 'app', 'css', 'styles.css')

with open(css_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find where the corruption starts: "[data-theme="dark"] .custom-multiselect-option input[type="checkbox"]:checked + label {"
start_idx = -1
for i, line in enumerate(lines):
    if '[data-theme="dark"] .custom-multiselect-option input[type="checkbox"]:checked + label' in line:
        start_idx = i
        break

if start_idx != -1:
    correct_end = """[data-theme="dark"] .custom-multiselect-option input[type="checkbox"]:checked + label {
    color: #e2e8f0;
}

/* 22. ANIMACIONES GLOBALES */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* 23. MOBILE-FIRST RESPONSIVE OVERRIDES */
@media (max-width: 768px) {
    .header-content {
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 1rem;
        gap: 0.8rem;
    }
    
    .header-right {
        justify-content: center;
    }

    .app-container, .container-fluid {
        padding: 0.5rem !important;
    }

    .nav-tabs {
        flex-wrap: wrap !important;
        justify-content: center !important;
        gap: 0.5rem;
        padding-bottom: 0;
    }
    
    .nav-link {
        flex: 1 1 45%;
        justify-content: center;
        text-align: center;
        font-size: 0.9rem;
        padding: 0.6rem;
    }

    .section-heading-inline {
        flex-direction: column;
        align-items: flex-start !important;
        gap: 0.3rem !important;
    }

    .layout-split-8-4 {
        grid-template-columns: 1fr !important;
        gap: 1rem;
    }

    .chat-container {
        height: 55vh !important;
    }

    .chat-input, .btn-primary {
        font-size: 16px !important;
    }
    
    .btn-primary {
        min-height: 48px;
        padding: 0.8rem 1rem;
        width: 100%;
        justify-content: center;
    }
    
    .chat-input-container {
        flex-direction: column;
    }
}
"""
    # Overwrite from start_idx to the end of the file with the correct_end
    with open(css_path, 'w', encoding='utf-8') as f:
        f.writelines(lines[:start_idx])
        f.write(correct_end)
    print("Fixed!")
else:
    print("Could not find start index")

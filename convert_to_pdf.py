import markdown
from weasyprint import HTML
import os

def convert_md_to_pdf(md_file, pdf_file):
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'codehilite']
    )
    
    # Add CSS styling
    html_with_style = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 40px;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
                margin-top: 30px;
            }}
            code {{
                background-color: #f8f9fa;
                padding: 2px 5px;
                border-radius: 3px;
                font-family: 'Courier New', Courier, monospace;
            }}
            pre {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            ul, ol {{
                margin-left: 20px;
            }}
            li {{
                margin-bottom: 5px;
            }}
            a {{
                color: #3498db;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f8f9fa;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert HTML to PDF
    HTML(string=html_with_style).write_pdf(pdf_file)
    
    print(f"PDF generated successfully: {pdf_file}")

if __name__ == "__main__":
    convert_md_to_pdf('report.md', 'emotion_recognition_report.pdf') 
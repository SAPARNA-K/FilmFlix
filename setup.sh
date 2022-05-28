
mkdir -p ~/.streamlit/

echo "\
[theme]\n\
primaryColor = '#F63366'\n\
backgroundColor = ''\n\
secondaryBackgroundColor = '#F0F2F6'\n\
textColor = '#dc3545'\n\
font = 'sans serif'\n\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS=false\n\
\n\
" > ~/.streamlit/config.toml


mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"20euai033@skcet.ac.in\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
port = $PORT\n\
\n\
" > ~/.streamlit/config.toml


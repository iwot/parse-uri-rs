#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Loc(usize, usize);

impl Loc {
    fn merge(&self, other: &Self) -> Self {
        use std::cmp::{max, min};
        Loc(min(self.0, other.0), max(self.1, other.1))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Annot<T> {
    value: T,
    loc: Loc,
}

impl<T> Annot<T> {
    fn new(value: T, loc: Loc) -> Self {
        Self { value, loc }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TokenKind {
    Authority(String, String, Option<String>, Option<String>),
    Scheme(String),
    // SubDelim(char),
    // GenDelim(char),
    // PctEncoded(String),
    Path(String),
    Query(String),
    Fragment(String),
}

type Token = Annot<TokenKind>;

impl Token {
    fn scheme(v: String, loc: Loc) -> Self {
        Self::new(TokenKind::Scheme(v), loc)
    }

    fn authority(v: String, host: String, user_info: Option<String>, port: Option<String>, loc: Loc) -> Self {
        Self::new(TokenKind::Authority(v, host, user_info, port), loc)
    }

    fn path(v: String, loc: Loc) -> Self {
        Self::new(TokenKind::Path(v), loc)
    }

    fn query(v: String, loc: Loc) -> Self {
        Self::new(TokenKind::Query(v), loc)
    }

    fn fragment(v: String, loc: Loc) -> Self {
        Self::new(TokenKind::Fragment(v), loc)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum LexErrorKind {
    InvalidChar(char),
    Eof,
}

type LexError = Annot<LexErrorKind>;

impl LexError {
    fn invalid_char(c: char, loc: Loc) -> Self {
        LexError::new(LexErrorKind::InvalidChar(c), loc)
    }
    fn eof (loc: Loc) -> Self {
        LexError::new(LexErrorKind::Eof, loc)
    }
}

fn tokens_to_uri(tokens: Vec<Token>) -> Option<super::uri::URI> {
    let mut scheme = String::new();
    let mut authority = String::new();
    let mut path = String::new();
    let mut fragment = String::new();
    let mut query = String::new();
    
    for t in tokens {
        match t.value {
            TokenKind::Scheme(a) => scheme = a,
            TokenKind::Authority(a, _, _, _) => authority = a,
            TokenKind::Path(a) => path = a,
            TokenKind::Fragment(a) => fragment = a,
            TokenKind::Query(a) => query = a,
        }
    }

    if scheme.len() == 0 || authority.len() == 0 {
        None
    } else {
        Some(super::uri::URI::new(&scheme, &authority, &path, &fragment, &query))
    }
}

fn lex(input: &str) -> Result<(Vec<Token>, Vec<super::uri::URI>), LexError> {
    let mut tokens = vec![];
    let mut result_tokens = vec![];
    let mut uris = vec![];
    // let input = input.as_bytes();
    let input = input.chars().collect::<Vec<char>>();
    let mut pos = 0;
    macro_rules! lex_a_token {
        ($lexer:expr) => {{
            let (tok, p) = $lexer?;
            tokens.push(tok);
            pos = p;
        }};
    }
    macro_rules! lex_a_tokens {
        ($lexer:expr) => {{
            let (toks, p) = $lexer?;
            tokens.extend(toks);
            pos = p;
        }};
    }
    // let hexdig_test = b"0123456789ABCDEFabcdef";

    let mut next_target = "scheme";
    while pos < input.len() {
        // println!("input[{}] = {:?}", pos, input[pos] as char);
        match input[pos] {
            'a'..='z' | 'A'..='Z' if next_target == "scheme" => {
                if tokens.len() > 0 {
                    if let Some(uri) = tokens_to_uri(tokens.clone()) {
                        uris.push(uri);
                        result_tokens.extend(tokens);
                    }
                    tokens = vec![];
                }
                lex_a_token!(lex_scheme(&input, pos));
                next_target = if input.len() > pos {
                    match &input[pos] {
                        ':' => "hier-part",
                        _ => "scheme",
                    }
                } else {
                    ""
                }
            },
            ':' if pos+3 < input.len() && next_target == "hier-part" => {
                lex_a_tokens!(lex_hier_part(&input, pos+3));
                next_target = if input.len() > pos {
                    match &input[pos] {
                        '?' => "query",
                        '#' => "fragment",
                        _ => "scheme",
                    }
                } else {
                    ""
                }
            },
            '?' if next_target == "query" => {
                lex_a_token!(lex_query(&input, pos+1));
                next_target = if input.len() > pos {
                    match &input[pos] {
                        '#' => "fragment",
                        _ => "scheme",
                    }
                } else {
                    ""
                }
            },
            '#' if next_target == "fragment" => {
                lex_a_token!(lex_fragment(&input, pos+1));
                next_target = "scheme";
            },
            ' ' | '\n' | '\t' if next_target != "hier-part" && next_target != "query-or-fragment" && next_target != "fragment" => {
                let((), p) = skip_spaces(&input, pos)?;
                pos = p;
                next_target = "scheme";
            },
            // b => return Err(LexError::invalid_char(b as char, Loc(pos, pos + 1))),
            _b => {
                pos += 1;
            },
        }
    }
    if tokens.len() > 0 {
        if let Some(uri) = tokens_to_uri(tokens.clone()) {
            uris.push(uri);
            result_tokens.extend(tokens);
        }
    }
    Ok((result_tokens, uris))
}

fn consume_byte(input: &Vec<char>, pos: usize, b: char) -> Result<(char, usize), LexError> {
    if input.len() <= pos {
        return Err(LexError::eof(Loc(pos, pos)));
    }

    // 入力が期待するものでない場合、エラー
    if input[pos] != b {
        return Err(LexError::invalid_char(input[pos] as char, Loc(pos, pos+1)));
    }

    Ok((b, pos))
}

// 続く限り認識
fn recognize_many0(input: &Vec<char>, mut pos: usize, mut f: impl FnMut(char) -> bool) -> usize {
    while pos < input.len() && f(input[pos]) {
        pos += 1;
    }
    pos
}

// 指定されたポジションまで認識
fn recognize_until_pos(
    input: &Vec<char>,
    mut pos: usize,
    until_pos: usize,
    mut f: impl FnMut(char) -> bool
) -> Result<usize, LexError> {
    while pos < input.len() && pos < until_pos {
        if !f(input[pos]) {
            return Err(LexError::invalid_char(input[pos] as char, Loc(pos, pos+1)));
        }
        pos += 1;
    }
    Ok(pos)
}

fn skip_spaces(input: &Vec<char>, pos: usize) -> Result<((), usize), LexError> {
    let pos = recognize_many0(input, pos, |b| " \t\n".chars().collect::<Vec<char>>().contains(&b));
    Ok(((), pos))
}

fn lex_scheme(input: &Vec<char>, pos: usize) -> Result<(Token, usize), LexError> {
    let start = pos;
    let end = recognize_many0(input, pos+1, |b| {
        ('a'..='z').contains(&b) || ('A'..='Z').contains(&b) || ('0'..='9').contains(&b) || "+-.".chars().collect::<Vec<char>>().contains(&b)
    });
    let data = input[start..end].iter().collect::<String>();
    Ok((Token::scheme(data.to_string(), Loc(start, end)), end))
}

fn lex_hier_part(input: &Vec<char>, pos: usize) -> Result<(Vec<Token>, usize), LexError> {
    let mut result = vec![];
    let mut result_pos;

    let(tok, pos) = lex_authority(input, pos)?;
    result.push(tok);
    result_pos = pos;

    if input.len() > pos {
        let(tok, pos) = lex_path(input, pos)?;
        result.push(tok);
        result_pos = pos;
    }

    // let(tok, pos) = lex_path_absolute(input, pos)?;
    // result.push(tok);

    // let(tok, pos) = lex_path_rootless(input, pos)?;
    // result.push(tok);

    // let(tok, pos) = lex_path_empty(input, pos)?;
    // result.push(tok);

    Ok((result, result_pos))
}

fn lex_authority(input: &Vec<char>, pos: usize) -> Result<(Token, usize), LexError> {
    use std::str::from_utf8;

    // @までがuserinfo、:から後がport
    // スラッシュ,空白,?,#の位置を見つける
    let mut last_pos = pos;
    while last_pos < input.len() {
        if "/ ?#\r\n".chars().collect::<Vec<char>>().contains(&input[last_pos]) {
            break;
        }
        last_pos += 1;
    }

    let mut at_mark_pos = None;
    let mut count_pos = pos;
    while count_pos < last_pos {
        if '@' == input[count_pos] {
            at_mark_pos = Some(count_pos);
            break;
        }
        count_pos += 1;
    }

    let host_start = if let Some(host_start) = at_mark_pos {
        host_start + 1
    } else {
        pos
    };

    let mut count_pos = host_start;
    while count_pos < last_pos {
        if ':' == input[count_pos] {
            break;
        }
        count_pos += 1;
    }
    let host_end = count_pos;

    let user_info = if let Some(at_mark_pos) = at_mark_pos {
        Some(input[pos..at_mark_pos].iter().collect::<String>())
    } else {
        None
    };

    let port = if last_pos > host_end + 1 {
        Some(input[(host_end+1)..last_pos].iter().collect::<String>())
    } else {
        None
    };

    // let data = from_utf8(&input[pos..last_pos]).unwrap();
    // let host = from_utf8(&input[host_start..host_end]).unwrap();
    let data = input[pos..last_pos].iter().collect::<String>();
    let host = input[host_start..host_end].iter().collect::<String>();
    Ok((Token::authority(data.to_string(), host.to_string(), user_info, port, Loc(pos, last_pos)), last_pos))
}

// pos以降のsearchを探す。
fn find_next_char_pos(input: &Vec<char>, pos: usize, search: char) -> Option<usize> {
    let mut current = pos;
    while current < input.len() {
        if search == input[current] {
            return Some(current);
        }

        current += 1;
    }
    None
}

// pos以降のsearchを探すが、stoppersを超えない。
fn find_next_char_pos_with_stoppers(input: &Vec<char>, pos: usize, search: char, stoppers: &Vec<char>) -> Option<usize> {
    let mut current = pos;
    while current < input.len() {
        if stoppers.contains(&input[current]) {
            break;
        }
        if search == input[current] {
            return Some(current);
        }

        current += 1;
    }
    None
}

fn find_next_chars_pos(input: &Vec<char>, pos: usize, search: &Vec<char>) -> Option<usize> {
    let mut current = pos;
    while current < input.len() {
        if search.contains(&input[current]) {
            return Some(current);
        }

        current += 1;
    }
    None
}

fn lex_path(input: &Vec<char>, pos: usize) -> Result<(Token, usize), LexError> {
    if '/' != input[pos] {
        return Err(LexError::invalid_char(input[pos] as char, Loc(pos, pos+1)));
    }

    let mut current = pos;
    if pos+1 < input.len() {
        if let Some(slash_pos) = find_next_char_pos_with_stoppers(input, current+1, '/', &"?# \r\n".chars().collect::<Vec<char>>()) {
            let (tok, new_pos) = lex_path(input, slash_pos)?;
            current = new_pos;
        } else {
            while current < input.len() {
                if "?# \r\n".chars().collect::<Vec<char>>().contains(&input[current]) {
                    break;
                }
                current += 1;
            }
        }
    }

    let data = input[pos..current].iter().collect::<String>();
    Ok((Token::path(data.to_string(), Loc(pos, current)), current))
}

fn lex_query(input: &Vec<char>, pos: usize) -> Result<(Token, usize), LexError> {
    let start = pos;
    let end = recognize_many0(input, start, |b| {
        ! " \r\n#[]".chars().collect::<Vec<char>>().contains(&b)
    });
    let data = input[start..end].iter().collect::<String>();
    Ok((Token::query(data.to_string(), Loc(start, end)), end))
}

fn lex_fragment(input: &Vec<char>, pos: usize) -> Result<(Token, usize), LexError> {
    let start = pos;
    let end = recognize_many0(input, start, |b| {
        ! " \r\n#[]".chars().collect::<Vec<char>>().contains(&b)
    });
    let data = input[start..end].iter().collect::<String>();
    Ok((Token::fragment(data.to_string(), Loc(start, end)), end))
}

pub fn extract_urls(text: &str) -> Vec<String> {
    let result = lex(text);
    if result.is_ok() {
        result.unwrap().1.iter().map(|p| p.raw()).collect()
    } else {
        vec![]
    }
}

#[test]
fn test_extract_urls() {
    let input = "あいうえお abc.ABC\r\nhttps://ccm.net/forum/affich-12027-finding-a-computer-name-using-ip\r\nかきくけこ https://github.com/iwot/parse-uri-rs
    ";
    let urls = extract_urls(input);
    assert_eq!(urls.len(), 2);
    assert_eq!(urls[0], "https://ccm.net/forum/affich-12027-finding-a-computer-name-using-ip");
    assert_eq!(urls[1], "https://github.com/iwot/parse-uri-rs");

    let input = "あいうえお abc.ABC\nhttps://ccm.net/forum/affich-12027-finding-a-computer-name-using-ip\nかきくけこ https://github.com/iwot/parse-uri-rs
    ";
    let urls = extract_urls(input);
    assert_eq!(urls.len(), 2);
    assert_eq!(urls[0], "https://ccm.net/forum/affich-12027-finding-a-computer-name-using-ip");
    assert_eq!(urls[1], "https://github.com/iwot/parse-uri-rs");
}

#[test]
fn test_recognize_until_pos() {
    let input = "01AB".chars().collect::<Vec<char>>();
    let result = recognize_until_pos(&input, 1, 3, |b| "0123456789ABCDEFabcdef".chars().collect::<Vec<char>>().contains(&b));
    // println!("{:?}", String::from_utf8(input[1..3].to_vec()));
    assert_eq!(result, Ok(3));
}

#[test]
fn test_lexer() {
    let expect = Ok((
        vec![
            Token::scheme("https".to_string(), Loc(1, 6)),
            Token::authority(
                "triple-underscore.github.io".to_string(),
                "triple-underscore.github.io".to_string(),
                None,
                None,
                Loc(9, 36)),
            Token::path("/rfc-others/RFC3986-ja.html".to_string(), Loc(36, 63)),
        ],
        vec![
            super::uri::URI::new("https", "triple-underscore.github.io", "/rfc-others/RFC3986-ja.html", "", ""),
        ]
    ));
    assert_eq!(
        lex(" https://triple-underscore.github.io/rfc-others/RFC3986-ja.html "),
        expect);
}

#[test]
fn test_lex_authority() {
    let input = "user@192.168.121.5:8080".chars().collect::<Vec<char>>();
    let expect = Token::authority(
        "user@192.168.121.5:8080".to_string(),
        "192.168.121.5".to_string(),
        Some("user".to_string()),
        Some("8080".to_string()),
        Loc(0, 23));
    assert_eq!(lex_authority(&input, 0), Ok((expect, 23)));
}

#[test]
fn test_lex_path() {
    let input = "/wiki/ABNF".chars().collect::<Vec<char>>();
    let expect = Token::path(
        "/wiki/ABNF".to_string(),
        Loc(0, 10));
    assert_eq!(lex_path(&input, 0), Ok((expect, 10)));

    let input = "/abc/def?hi=jk".chars().collect::<Vec<char>>();
    let expect = Token::path(
        "/abc/def".to_string(),
        Loc(0, 8));
    assert_eq!(lex_path(&input, 0), Ok((expect, 8)));
}



#[test]
fn test_lex_query() {
    let input = "abcdef=aaaaaa&aaa=123#test".chars().collect::<Vec<char>>();
    let expect = Token::query(
        "abcdef=aaaaaa&aaa=123".to_string(),
        Loc(0, 21));
    assert_eq!(lex_query(&input, 0), Ok((expect, 21)));
}

#[test]
fn test_lex_fragment() {
    let input = "aaa=123#test".chars().collect::<Vec<char>>();
    let expect = Token::fragment(
        "test".to_string(),
        Loc(8, 12));
    assert_eq!(lex_fragment(&input, 8), Ok((expect, 12)));
}

#[test]
fn test_lex_hier_part() {
    let input = "http://user@zxy.com:8080/aa1/bb2/cc3".chars().collect::<Vec<char>>();
    let result = lex_hier_part(&input, 7);
    let expect_authority = Token::authority(
        "user@zxy.com:8080".to_string(),
        "zxy.com".to_string(),
        Some("user".to_string()),
        Some("8080".to_string()),
        Loc(7, 24));
    let expect_path = Token::path("/aa1/bb2/cc3".to_string(), Loc(24, 36));
    let expect = vec![expect_authority, expect_path];
    assert_eq!(result, Ok((expect, 36)));
}

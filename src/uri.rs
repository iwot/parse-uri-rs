#[derive(Debug, PartialEq)]
pub struct URI {
    raw: String,
    scheme: String,
    authority: String,
    path: String,
    query: String,
    fragment: String,
}

impl URI {
    pub fn new(scheme: &str, authority: &str, path: &str, query: &str, fragment: &str) -> Self {
        let mut raw = String::new();
        raw += scheme.clone();
        raw += "://";
        raw += authority.clone();
        raw += path.clone();
        if query.len() > 0 {
            raw += "?";
            raw += query.clone();
        }
        if fragment.len() > 0 {
            raw += "#";
            raw += fragment.clone();
        }

        let scheme = scheme.to_string();
        let authority = authority.to_string();
        let path = path.to_string();
        let query = query.to_string();
        let fragment = fragment.to_string();
        Self {raw, scheme, authority, path, query, fragment}
    }

    pub fn raw(&self) -> String {
        self.raw.clone()
    }

    pub fn scheme(&self) -> String {
        self.scheme.clone()
    }

    pub fn authority(&self) -> String {
        self.authority.clone()
    }

    pub fn path(&self) -> String {
        self.path.clone()
    }

    pub fn query(&self) -> String {
        self.query.clone()
    }

    pub fn fragment(&self) -> String {
        self.fragment.clone()
    }
}


"""
Token Collector for Go Code

Collects tokenized Go code using the go/scanner package via subprocess.
Stores token sequences with position info for training.

Key Features:
- Uses Go's actual scanner for authentic tokenization
- Preserves position info for alignment
- Creates train/test splits from stdlib and sample repos
- Supports streaming large codebases
"""

import subprocess
import json
import tempfile
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterator, Tuple
from pathlib import Path
import random
from enum import Enum


class GoTokenType(Enum):
    """Go token types from go/token package."""
    # Special tokens
    ILLEGAL = "ILLEGAL"
    EOF = "EOF"
    COMMENT = "COMMENT"
    
    # Identifiers and literals
    IDENT = "IDENT"
    INT = "INT"
    FLOAT = "FLOAT"
    IMAG = "IMAG"
    CHAR = "CHAR"
    STRING = "STRING"
    
    # Operators and delimiters
    ADD = "+"
    SUB = "-"
    MUL = "*"
    QUO = "/"
    REM = "%"
    AND = "&"
    OR = "|"
    XOR = "^"
    SHL = "<<"
    SHR = ">>"
    AND_NOT = "&^"
    ADD_ASSIGN = "+="
    SUB_ASSIGN = "-="
    MUL_ASSIGN = "*="
    QUO_ASSIGN = "/="
    REM_ASSIGN = "%="
    AND_ASSIGN = "&="
    OR_ASSIGN = "|="
    XOR_ASSIGN = "^="
    SHL_ASSIGN = "<<="
    SHR_ASSIGN = ">>="
    AND_NOT_ASSIGN = "&^="
    LAND = "&&"
    LOR = "||"
    ARROW = "<-"
    INC = "++"
    DEC = "--"
    EQL = "=="
    LSS = "<"
    GTR = ">"
    ASSIGN = "="
    NOT = "!"
    NEQ = "!="
    LEQ = "<="
    GEQ = ">="
    DEFINE = ":="
    ELLIPSIS = "..."
    LPAREN = "("
    LBRACK = "["
    LBRACE = "{"
    COMMA = ","
    PERIOD = "."
    RPAREN = ")"
    RBRACK = "]"
    RBRACE = "}"
    SEMICOLON = ";"
    COLON = ":"
    
    # Keywords
    BREAK = "break"
    CASE = "case"
    CHAN = "chan"
    CONST = "const"
    CONTINUE = "continue"
    DEFAULT = "default"
    DEFER = "defer"
    ELSE = "else"
    FALLTHROUGH = "fallthrough"
    FOR = "for"
    FUNC = "func"
    GO = "go"
    GOTO = "goto"
    IF = "if"
    IMPORT = "import"
    INTERFACE = "interface"
    MAP = "map"
    PACKAGE = "package"
    RANGE = "range"
    RETURN = "return"
    SELECT = "select"
    STRUCT = "struct"
    SWITCH = "switch"
    TYPE = "type"
    VAR = "var"


# Token type to ID mapping for model training
TOKEN_VOCAB = {t.value: i for i, t in enumerate(GoTokenType)}
TOKEN_VOCAB['<PAD>'] = len(TOKEN_VOCAB)
TOKEN_VOCAB['<UNK>'] = len(TOKEN_VOCAB)
TOKEN_VOCAB['<BOS>'] = len(TOKEN_VOCAB)
TOKEN_VOCAB['<EOS>'] = len(TOKEN_VOCAB)


@dataclass
class GoToken:
    """A single Go token with position information."""
    type_name: str       # Token type name (e.g., "IDENT", "FUNC")
    literal: str         # Actual text (e.g., "main", "func")
    line: int           # Line number (1-indexed)
    column: int         # Column number (1-indexed)
    offset: int         # Byte offset in file
    
    @property
    def type_id(self) -> int:
        """Get vocabulary ID for this token type."""
        return TOKEN_VOCAB.get(self.type_name, TOKEN_VOCAB['<UNK>'])
    
    @property
    def literal_id(self) -> int:
        """Get vocabulary ID for literal (using type as fallback)."""
        # For keywords and operators, use the type
        if self.type_name in TOKEN_VOCAB:
            return TOKEN_VOCAB[self.type_name]
        # For identifiers and literals, hash to a range
        return hash(self.literal) % 10000 + len(TOKEN_VOCAB)


@dataclass
class TokenSequence:
    """A sequence of tokens from a Go source file."""
    tokens: List[GoToken]
    file_path: str
    package_name: str = ""
    is_test: bool = False
    
    def __len__(self) -> int:
        return len(self.tokens)
    
    def to_type_ids(self) -> List[int]:
        """Convert to sequence of type IDs."""
        return [t.type_id for t in self.tokens]
    
    def to_literal_ids(self) -> List[int]:
        """Convert to sequence of literal IDs."""
        return [t.literal_id for t in self.tokens]
    
    def ngrams(self, n: int) -> Iterator[List[GoToken]]:
        """Generate n-grams of tokens."""
        for i in range(len(self.tokens) - n + 1):
            yield self.tokens[i:i+n]
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'file_path': self.file_path,
            'package_name': self.package_name,
            'is_test': self.is_test,
            'tokens': [
                {
                    'type': t.type_name,
                    'lit': t.literal,
                    'line': t.line,
                    'col': t.column,
                    'off': t.offset,
                }
                for t in self.tokens
            ]
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TokenSequence':
        """Deserialize from dictionary."""
        tokens = [
            GoToken(
                type_name=t['type'],
                literal=t['lit'],
                line=t['line'],
                column=t['col'],
                offset=t['off'],
            )
            for t in d['tokens']
        ]
        return cls(
            tokens=tokens,
            file_path=d['file_path'],
            package_name=d.get('package_name', ''),
            is_test=d.get('is_test', False),
        )


# Go program to tokenize files
TOKENIZER_GO_CODE = '''
package main

import (
    "encoding/json"
    "fmt"
    "go/scanner"
    "go/token"
    "os"
)

type Token struct {
    Type   string `json:"type"`
    Lit    string `json:"lit"`
    Line   int    `json:"line"`
    Col    int    `json:"col"`
    Offset int    `json:"off"`
}

func main() {
    if len(os.Args) < 2 {
        fmt.Fprintln(os.Stderr, "usage: tokenizer <file>")
        os.Exit(1)
    }

    src, err := os.ReadFile(os.Args[1])
    if err != nil {
        fmt.Fprintf(os.Stderr, "error reading file: %v\\n", err)
        os.Exit(1)
    }

    fset := token.NewFileSet()
    file := fset.AddFile(os.Args[1], fset.Base(), len(src))

    var s scanner.Scanner
    s.Init(file, src, nil, scanner.ScanComments)

    var tokens []Token
    for {
        pos, tok, lit := s.Scan()
        position := fset.Position(pos)
        
        tokens = append(tokens, Token{
            Type:   tok.String(),
            Lit:    lit,
            Line:   position.Line,
            Col:    position.Column,
            Offset: position.Offset,
        })

        if tok == token.EOF {
            break
        }
    }

    enc := json.NewEncoder(os.Stdout)
    enc.SetEscapeHTML(false)
    enc.Encode(tokens)
}
'''


class GoTokenCollector:
    """
    Collects tokenized Go code from files and directories.
    
    Uses go/scanner via subprocess for authentic tokenization.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix='go_tokens_')
        self.tokenizer_path = os.path.join(self.cache_dir, 'tokenizer')
        self._build_tokenizer()
        
        # Collected sequences
        self.sequences: List[TokenSequence] = []
    
    def _build_tokenizer(self):
        """Build the Go tokenizer binary."""
        go_file = os.path.join(self.cache_dir, 'tokenizer.go')
        with open(go_file, 'w') as f:
            f.write(TOKENIZER_GO_CODE)
        
        result = subprocess.run(
            ['go', 'build', '-o', self.tokenizer_path, go_file],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to build tokenizer: {result.stderr}")
    
    def tokenize_file(self, file_path: str) -> Optional[TokenSequence]:
        """
        Tokenize a single Go file.
        
        Returns TokenSequence or None if file cannot be tokenized.
        """
        if not os.path.exists(file_path):
            return None
        
        result = subprocess.run(
            [self.tokenizer_path, file_path],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            return None
        
        try:
            token_dicts = json.loads(result.stdout)
            tokens = [
                GoToken(
                    type_name=t['type'],
                    literal=t['lit'],
                    line=t['line'],
                    column=t['col'],
                    offset=t['off'],
                )
                for t in token_dicts
            ]
            
            # Extract package name from first IDENT after "package"
            package_name = ""
            for i, t in enumerate(tokens):
                if t.type_name == "package" and i + 1 < len(tokens):
                    package_name = tokens[i + 1].literal
                    break
            
            return TokenSequence(
                tokens=tokens,
                file_path=file_path,
                package_name=package_name,
                is_test='_test.go' in file_path,
            )
        except json.JSONDecodeError:
            return None
    
    def tokenize_directory(
        self,
        dir_path: str,
        recursive: bool = True,
        include_tests: bool = False,
        max_files: Optional[int] = None,
    ) -> List[TokenSequence]:
        """
        Tokenize all Go files in a directory.
        
        Args:
            dir_path: Directory to scan
            recursive: Whether to scan subdirectories
            include_tests: Whether to include test files
            max_files: Maximum number of files to process
        
        Returns:
            List of TokenSequence objects
        """
        sequences = []
        files_processed = 0
        
        path = Path(dir_path)
        pattern = '**/*.go' if recursive else '*.go'
        
        for go_file in path.glob(pattern):
            if max_files and files_processed >= max_files:
                break
            
            if not include_tests and '_test.go' in str(go_file):
                continue
            
            # Skip vendor and testdata directories
            if 'vendor' in str(go_file) or 'testdata' in str(go_file):
                continue
            
            seq = self.tokenize_file(str(go_file))
            if seq:
                sequences.append(seq)
                files_processed += 1
        
        return sequences
    
    def tokenize_stdlib(
        self,
        goroot: Optional[str] = None,
        packages: Optional[List[str]] = None,
        max_files_per_pkg: int = 50,
    ) -> List[TokenSequence]:
        """
        Tokenize Go standard library.
        
        Args:
            goroot: GOROOT path (defaults to `go env GOROOT`)
            packages: Specific packages to tokenize (defaults to common ones)
            max_files_per_pkg: Maximum files per package
        
        Returns:
            List of TokenSequence objects
        """
        if goroot is None:
            result = subprocess.run(
                ['go', 'env', 'GOROOT'],
                capture_output=True,
                text=True,
            )
            goroot = result.stdout.strip()
        
        if packages is None:
            packages = [
                'fmt', 'io', 'os', 'net', 'http', 'strings', 'bytes',
                'strconv', 'sort', 'sync', 'context', 'errors', 'time',
                'encoding/json', 'encoding/xml', 'path/filepath',
                'regexp', 'bufio', 'container/list', 'container/heap',
            ]
        
        sequences = []
        src_dir = os.path.join(goroot, 'src')
        
        for pkg in packages:
            pkg_dir = os.path.join(src_dir, pkg)
            if os.path.exists(pkg_dir):
                pkg_seqs = self.tokenize_directory(
                    pkg_dir,
                    recursive=False,
                    include_tests=False,
                    max_files=max_files_per_pkg,
                )
                sequences.extend(pkg_seqs)
        
        return sequences
    
    def collect(
        self,
        sources: List[str],
        include_stdlib: bool = True,
        max_total_files: int = 1000,
    ) -> 'GoTokenCollector':
        """
        Collect tokens from multiple sources.
        
        Args:
            sources: List of file/directory paths
            include_stdlib: Whether to include stdlib
            max_total_files: Maximum total files
        
        Returns:
            self for chaining
        """
        if include_stdlib:
            stdlib_seqs = self.tokenize_stdlib(
                max_files_per_pkg=max_total_files // 20
            )
            self.sequences.extend(stdlib_seqs)
        
        remaining = max_total_files - len(self.sequences)
        
        for source in sources:
            if remaining <= 0:
                break
            
            if os.path.isfile(source):
                seq = self.tokenize_file(source)
                if seq:
                    self.sequences.append(seq)
                    remaining -= 1
            elif os.path.isdir(source):
                seqs = self.tokenize_directory(
                    source,
                    max_files=remaining,
                )
                self.sequences.extend(seqs)
                remaining -= len(seqs)
        
        return self
    
    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[TokenSequence], List[TokenSequence], List[TokenSequence]]:
        """
        Split sequences into train/val/test sets.
        
        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            seed: Random seed
        
        Returns:
            (train, val, test) tuple of sequence lists
        """
        random.seed(seed)
        sequences = self.sequences.copy()
        random.shuffle(sequences)
        
        n = len(sequences)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        return (
            sequences[:train_end],
            sequences[train_end:val_end],
            sequences[val_end:],
        )
    
    def get_transition_counts(self) -> Dict[Tuple[str, str], int]:
        """
        Count token type bigram transitions.
        
        Returns:
            Dict mapping (from_type, to_type) to count
        """
        counts: Dict[Tuple[str, str], int] = {}
        
        for seq in self.sequences:
            for i in range(len(seq.tokens) - 1):
                from_type = seq.tokens[i].type_name
                to_type = seq.tokens[i + 1].type_name
                key = (from_type, to_type)
                counts[key] = counts.get(key, 0) + 1
        
        return counts
    
    def get_token_stats(self) -> Dict[str, int]:
        """Get token type frequency statistics."""
        stats: Dict[str, int] = {}
        for seq in self.sequences:
            for token in seq.tokens:
                stats[token.type_name] = stats.get(token.type_name, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: -x[1]))
    
    def save(self, path: str):
        """Save collected sequences to JSON file."""
        data = [seq.to_dict() for seq in self.sequences]
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str) -> 'GoTokenCollector':
        """Load sequences from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.sequences = [TokenSequence.from_dict(d) for d in data]
        return self


def demo():
    """Demonstrate token collection."""
    print("=" * 60)
    print("GO TOKEN COLLECTOR DEMO")
    print("=" * 60)
    
    collector = GoTokenCollector()
    
    # Create a sample Go file
    sample_code = '''
package main

import "fmt"

func main() {
    x := 42
    if x > 0 {
        fmt.Println("positive")
    }
}
'''
    
    sample_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.go', delete=False
    )
    sample_file.write(sample_code)
    sample_file.close()
    
    print(f"\nTokenizing sample file: {sample_file.name}")
    seq = collector.tokenize_file(sample_file.name)
    
    if seq:
        print(f"\nFound {len(seq)} tokens:")
        for i, token in enumerate(seq.tokens[:20]):
            print(f"  {i:3d}: {token.type_name:12s} | {token.literal!r:15s} | "
                  f"line {token.line}, col {token.column}")
        
        if len(seq.tokens) > 20:
            print(f"  ... and {len(seq.tokens) - 20} more tokens")
        
        print(f"\nToken type IDs (first 10): {seq.to_type_ids()[:10]}")
        
        print("\nBigram transitions:")
        for ngram in list(seq.ngrams(2))[:5]:
            print(f"  {ngram[0].type_name} -> {ngram[1].type_name}")
    
    # Cleanup
    os.unlink(sample_file.name)
    
    print("\n" + "=" * 60)
    print("STDLIB TOKENIZATION")
    print("=" * 60)
    
    # Tokenize a bit of stdlib
    stdlib_seqs = collector.tokenize_stdlib(
        packages=['fmt'],
        max_files_per_pkg=3,
    )
    
    print(f"\nTokenized {len(stdlib_seqs)} files from fmt package")
    
    if stdlib_seqs:
        stats = {}
        for seq in stdlib_seqs:
            for token in seq.tokens:
                stats[token.type_name] = stats.get(token.type_name, 0) + 1
        
        print("\nTop 10 token types:")
        for typ, count in sorted(stats.items(), key=lambda x: -x[1])[:10]:
            print(f"  {typ:15s}: {count:5d}")


if __name__ == "__main__":
    demo()

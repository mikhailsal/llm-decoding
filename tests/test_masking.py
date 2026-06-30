from dsbx.web.logging.masking import mask_api_key, mask_headers, mask_sensitive_fields


def test_mask_api_key():
    assert mask_api_key("") == "***"
    assert mask_api_key(None) == "***"
    assert mask_api_key("123") == "***"
    assert mask_api_key("123456") == "***"
    assert mask_api_key("1234567") == "123*567"
    assert mask_api_key("abcdefgh") == "abc**fgh"


def test_mask_headers():
    assert mask_headers(None) is None
    headers = {
        "Authorization": "Bearer key1234567",
        "Content-Type": "application/json",
        "X-Api-Key": "my-secret-key",
        "X-Some-Header": "public-value",
        "password": "pass",
        "empty-token": "",
    }
    expected = {
        "Authorization": "Bea***********567",
        "Content-Type": "application/json",
        "X-Api-Key": "my-*******key",
        "X-Some-Header": "public-value",
        "password": "***",
        "empty-token": "",
    }
    assert mask_headers(headers) == expected


def test_mask_sensitive_fields():
    assert mask_sensitive_fields(None) is None
    assert mask_sensitive_fields(123) == 123
    assert mask_sensitive_fields("hello") == "hello"

    data = {
        "api_key": "secret123456",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "you are a helper"},
        ],
        "nested": {"password": "mypassword123", "non_sensitive": "value"},
        "list_of_secrets": [{"secret_token": "token12345"}, {"other": "public"}],
    }

    expected = {
        "api_key": "sec******456",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "you are a helper"},
        ],
        "nested": {"password": "myp*******123", "non_sensitive": "value"},
        "list_of_secrets": [{"secret_token": "tok****345"}, {"other": "public"}],
    }
    assert mask_sensitive_fields(data) == expected

from micrograd.engine import Value

def test_sanity_check():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    
    # Test forward pass
    assert abs(y.data - 20.0) < 1e-6
    # Test backward pass
    assert abs(x.grad - 46.0) < 1e-6

def test_more_ops():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    
    # Test forward pass
    assert abs(g.data - 24.70408163265306) < 1e-6
    # Test backward pass
    assert abs(a.grad - 138.83381924198252) < 1e-6
    assert abs(b.grad - 645.5772594752186) < 1e-6

if __name__ == "__main__":
    test_sanity_check()
    test_more_ops()
    print("All tests passed!") 
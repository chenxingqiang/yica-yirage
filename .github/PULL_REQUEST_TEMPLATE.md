# Pull Request

## 📝 Description
Brief description of what this PR does.

Fixes #(issue_number)

## 🔄 Type of Change
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] ⚡ Performance improvement
- [ ] 🧹 Code cleanup/refactoring
- [ ] 🔧 Build system changes
- [ ] 🧪 Test improvements

## 🎯 Backend Impact
Which backends are affected by this change?
- [ ] CPU backend
- [ ] CUDA backend  
- [ ] MPS backend
- [ ] Backend-agnostic changes
- [ ] New backend: ___________

## 🧪 Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] I have run the test suite locally and all tests pass
- [ ] I have tested on multiple backends (specify which ones)

**Test Results**:
```
# Paste test output here
python tools/yirage_backend_manager.py test --backend <backend>
```

## 📊 Performance Impact
- [ ] No performance impact expected
- [ ] Performance improvement (provide benchmarks)
- [ ] Potential performance regression (justified because ...)

**Benchmark Results** (if applicable):
```
# Before:
Backend: CPU, Avg: 45.2ms ± 2.1ms

# After:  
Backend: CPU, Avg: 38.7ms ± 1.8ms (14.4% improvement)
```

## 🔗 Dependencies
- [ ] This change requires a documentation update
- [ ] This change requires updating examples
- [ ] This change requires updating Docker images
- [ ] This change requires updating CI/CD pipelines

## 📚 Documentation
- [ ] I have updated relevant documentation
- [ ] I have added docstrings to new functions/classes
- [ ] I have updated the changelog
- [ ] I have added examples for new features

## ✅ Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## 🖼️ Screenshots (if applicable)
Add screenshots to help explain your changes.

## 📋 Additional Notes
Any additional information that reviewers should know.
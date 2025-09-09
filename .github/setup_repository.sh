#!/bin/bash
# GitHub Repository Setup Script for YiRage Multi-Backend
# Copyright 2025-2026 YICA TEAM

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    print_error "GitHub CLI (gh) is not installed. Please install it first:"
    echo "  https://github.com/cli/cli#installation"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    print_error "Please authenticate with GitHub CLI first:"
    echo "  gh auth login"
    exit 1
fi

REPO_OWNER="chenxingqiang"
REPO_NAME="yica-yirage"
REPO_FULL="${REPO_OWNER}/${REPO_NAME}"

print_info "Setting up GitHub repository: $REPO_FULL"

# Set repository settings
print_info "Configuring repository settings..."

# Enable discussions
gh api repos/$REPO_FULL -X PATCH -f has_discussions=true || print_warning "Could not enable discussions"

# Enable vulnerability alerts
gh api repos/$REPO_FULL -X PATCH -f security_and_analysis='{"vulnerability_alerts":{"status":"enabled"}}' || print_warning "Could not enable vulnerability alerts"

# Set default branch protection (if on main)
if git rev-parse --verify main &> /dev/null; then
    print_info "Setting up branch protection for main..."
    
    gh api repos/$REPO_FULL/branches/main/protection -X PUT \
        --input - << 'EOF' || print_warning "Could not set branch protection"
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["Quick Test (cpu)"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF
fi

# Create labels
print_info "Creating repository labels..."

# Define labels
declare -A LABELS=(
    ["bug"]="d73a4a"
    ["enhancement"]="a2eeef"
    ["performance"]="ff6600"
    ["documentation"]="0075ca"
    ["backend:cpu"]="b60205"
    ["backend:cuda"]="76B900"
    ["backend:mps"]="9932cc"
    ["priority:high"]="ff0000"
    ["priority:medium"]="ff9900"
    ["priority:low"]="008000"
    ["good first issue"]="7057ff"
    ["help wanted"]="008672"
    ["question"]="d876e3"
    ["wontfix"]="ffffff"
    ["duplicate"]="cfd3d7"
    ["invalid"]="e4e669"
    ["breaking change"]="B60205"
    ["ci/cd"]="1d76db"
    ["testing"]="0e8a16"
)

for label in "${!LABELS[@]}"; do
    color="${LABELS[$label]}"
    gh api repos/$REPO_FULL/labels -X POST \
        -f name="$label" \
        -f color="$color" \
        -f description="" \
        2>/dev/null || echo "Label '$label' might already exist"
done

print_success "Labels created"

# Create repository secrets for CI/CD (these would need to be set manually)
print_info "Repository secrets that should be configured:"
echo "  - DOCKER_HUB_USERNAME (for Docker image publishing)"
echo "  - DOCKER_HUB_ACCESS_TOKEN (for Docker image publishing)"
echo "  - CODECOV_TOKEN (for code coverage reporting)"
echo ""
echo "Set these with: gh secret set SECRET_NAME"

# Create initial issues for tracking
print_info "Creating initial project issues..."

# Create milestone for v1.0
gh api repos/$REPO_FULL/milestones -X POST \
    -f title="Multi-Backend v1.0" \
    -f description="Initial multi-backend release" \
    -f state="open" \
    2>/dev/null || print_warning "Could not create milestone"

# Initial issues
cat > /tmp/initial_issues.json << 'EOF'
[
  {
    "title": "📚 Update documentation for multi-backend support",
    "body": "Update all documentation to reflect the new multi-backend architecture.\n\n**Tasks:**\n- [ ] Update README.md\n- [ ] Update API documentation\n- [ ] Create migration guide\n- [ ] Add examples for each backend\n- [ ] Update Docker documentation",
    "labels": ["documentation", "enhancement"]
  },
  {
    "title": "🧪 Expand test coverage for all backends",
    "body": "Ensure comprehensive test coverage across CPU, CUDA, and MPS backends.\n\n**Tasks:**\n- [ ] Add CPU backend unit tests\n- [ ] Add CUDA backend unit tests\n- [ ] Add MPS backend unit tests\n- [ ] Add integration tests\n- [ ] Add performance regression tests",
    "labels": ["testing", "enhancement"]
  },
  {
    "title": "⚡ Performance optimization for CPU backend",
    "body": "Optimize CPU backend performance using advanced SIMD instructions and better memory management.\n\n**Goals:**\n- [ ] Implement AVX-512 kernels\n- [ ] Optimize memory layout\n- [ ] Add NUMA awareness\n- [ ] Benchmark against reference implementations",
    "labels": ["performance", "backend:cpu"]
  },
  {
    "title": "🐛 Fix MPS backend memory management",
    "body": "Address memory management issues in the MPS backend for better stability.\n\n**Issues:**\n- Memory leaks in buffer allocation\n- Incorrect memory alignment\n- Buffer reuse optimization\n\n**Priority:** High",
    "labels": ["bug", "backend:mps", "priority:high"]
  },
  {
    "title": "✨ Add support for AMD ROCm backend",
    "body": "Implement support for AMD GPUs using ROCm platform.\n\n**Requirements:**\n- [ ] Design ROCm backend interface\n- [ ] Implement HIP kernels\n- [ ] Add ROCm detection logic\n- [ ] Create ROCm Docker image\n- [ ] Add CI/CD support",
    "labels": ["enhancement", "good first issue"]
  }
]
EOF

# Create issues
python3 -c "
import json
import subprocess
import sys

with open('/tmp/initial_issues.json') as f:
    issues = json.load(f)

for issue in issues:
    cmd = [
        'gh', 'issue', 'create',
        '--title', issue['title'],
        '--body', issue['body'],
        '--repo', '$REPO_FULL'
    ]
    
    for label in issue['labels']:
        cmd.extend(['--label', label])
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f'✓ Created issue: {issue[\"title\"]}')
    except subprocess.CalledProcessError as e:
        print(f'✗ Failed to create issue: {issue[\"title\"]}')
        print(f'  Error: {e.stderr.decode()}')
"

rm -f /tmp/initial_issues.json

# Enable GitHub Pages (if docs exist)
if [ -d "docs" ]; then
    print_info "Enabling GitHub Pages for documentation..."
    gh api repos/$REPO_FULL/pages -X POST \
        -f source='{"branch":"main","path":"/docs"}' \
        2>/dev/null || print_warning "Could not enable GitHub Pages"
fi

# Set repository topics
print_info "Setting repository topics..."
gh api repos/$REPO_FULL -X PATCH \
    -f topics='["ai","machine-learning","deep-learning","llm","inference","optimization","cuda","cpu","apple-silicon","mps","pytorch","performance"]' \
    2>/dev/null || print_warning "Could not set repository topics"

print_success "Repository setup completed!"

echo ""
echo "Next steps:"
echo "1. Push your code to the repository:"
echo "   git add ."
echo "   git commit -m 'Initial multi-backend implementation'"
echo "   git push -u origin main"
echo ""
echo "2. Configure repository secrets for CI/CD"
echo "3. Review and adjust branch protection rules"
echo "4. Set up GitHub Pages for documentation"
echo "5. Configure GitHub Discussions categories"
echo ""
echo "Repository URL: https://github.com/$REPO_FULL"
echo "Actions URL: https://github.com/$REPO_FULL/actions"
echo "Issues URL: https://github.com/$REPO_FULL/issues"

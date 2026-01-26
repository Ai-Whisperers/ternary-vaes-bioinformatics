#!/usr/bin/env python3
"""
Email Outreach Sender - Professional HTML Emails

Sends beautifully formatted emails from markdown draft files via Gmail SMTP.

Usage:
    # Test mode - sends all "READY TO SEND" drafts to test email
    python send_emails.py --test --test-email your@email.com

    # Send a single draft (test mode)
    python send_emails.py --draft 01_brian_kuhlman.md --test --test-email your@email.com

    # REAL MODE - Actually sends to recipients (BE CAREFUL!)
    python send_emails.py --real --confirm

    # Send specific drafts in real mode
    python send_emails.py --draft 01_brian_kuhlman.md --real --confirm

Requirements:
    - Gmail account with App Password (not regular password)
    - Enable 2FA on Gmail, then create App Password at:
      https://myaccount.google.com/apppasswords

Environment Variables:
    GMAIL_ADDRESS: Your Gmail address (sender)
    GMAIL_APP_PASSWORD: Your Gmail App Password (16 characters, no spaces)
"""

import argparse
import html
import os
import re
import smtplib
import sys
import time
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional


@dataclass
class EmailDraft:
    """Parsed email draft from markdown file."""

    filename: str
    recipient_name: str
    recipient_email: str
    subject: str
    body: str
    status: str
    word_count: int


# Professional HTML email template
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <!--[if mso]>
  <style type="text/css">
    body, table, td {{font-family: Arial, Helvetica, sans-serif !important;}}
  </style>
  <![endif]-->
</head>
<body style="margin: 0; padding: 0; background-color: #f8f9fa; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; -webkit-font-smoothing: antialiased;">
  <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background-color: #f8f9fa;">
    <tr>
      <td style="padding: 24px 16px;">
        <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
          
          <!-- Email Body -->
          <tr>
            <td style="padding: 36px 40px 24px 40px;">
              <div style="color: #1a1a1a; font-size: 15px; line-height: 1.7;">
                {body_html}
              </div>
            </td>
          </tr>
          
          <!-- Signature -->
          <tr>
            <td style="padding: 0 40px 36px 40px;">
              <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                <tr>
                  <td style="border-top: 1px solid #e9ecef; padding-top: 24px;">
                    <p style="margin: 0 0 20px 0; color: #1a1a1a; font-size: 15px;">Best,</p>
                    
                    <!-- Team Names -->
                    <p style="margin: 0 0 2px 0; color: #1a1a1a; font-size: 15px; font-weight: 600;">Ivan Weiss Van Der Pol</p>
                    <p style="margin: 0 0 2px 0; color: #1a1a1a; font-size: 15px; font-weight: 600;">Jonathan Verdun</p>
                    <p style="margin: 0 0 16px 0; color: #1a1a1a; font-size: 15px; font-weight: 600;">Kyrian Weiss Van Der Pol</p>
                    
                    <!-- Organization -->
                    <p style="margin: 0 0 8px 0; color: #6c757d; font-size: 14px;">
                      <strong style="color: #2563eb;">AI Whisperers</strong>&nbsp;&nbsp;|&nbsp;&nbsp;CONACYT Paraguay
                    </p>
                    
                    <!-- GitHub Link -->
                    <p style="margin: 0;">
                      <a href="https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics" 
                         style="color: #2563eb; text-decoration: none; font-size: 14px;">
                         &#128279; github.com/Ai-Whisperers/ternary-vaes-bioinformatics
                      </a>
                    </p>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          
        </table>
        
        <!-- Footer -->
        <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="max-width: 600px; margin: 0 auto;">
          <tr>
            <td style="padding: 16px 40px; text-align: center;">
              <p style="margin: 0; color: #9ca3af; font-size: 12px;">
                P-adic Hyperbolic Variational Autoencoders for Bioinformatics
              </p>
            </td>
          </tr>
        </table>
        
      </td>
    </tr>
  </table>
</body>
</html>"""


def markdown_to_html(text: str) -> str:
    """Convert markdown-like text to HTML with proper formatting."""
    # Escape HTML first
    text = html.escape(text)

    # Convert line breaks to paragraphs
    paragraphs = text.split("\n\n")
    html_parts = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Skip signature (we handle it in template)
        if para.startswith("Best,") or "AI Whisperers" in para or "github.com" in para:
            continue
        if para in ["Ivan Weiss Van Der Pol", "Jonathan Verdun", "Kyrian Weiss Van Der Pol"]:
            continue

        # Handle numbered lists
        if re.match(r"^\d+\.", para):
            lines = para.split("\n")
            list_items = []
            for line in lines:
                line = line.strip()
                if re.match(r"^\d+\.", line):
                    item_text = re.sub(r"^\d+\.\s*", "", line)
                    list_items.append(f'<li style="margin-bottom: 8px;">{item_text}</li>')
            if list_items:
                html_parts.append(
                    f'<ol style="margin: 16px 0; padding-left: 24px; color: #1a1a1a;">{"".join(list_items)}</ol>'
                )
            continue

        # Handle bullet points
        if para.startswith("- ") or para.startswith("* "):
            lines = para.split("\n")
            list_items = []
            for line in lines:
                line = line.strip()
                if line.startswith("- ") or line.startswith("* "):
                    item_text = line[2:]
                    list_items.append(f'<li style="margin-bottom: 8px;">{item_text}</li>')
            if list_items:
                html_parts.append(
                    f'<ul style="margin: 16px 0; padding-left: 24px; color: #1a1a1a;">{"".join(list_items)}</ul>'
                )
            continue

        # Convert single line breaks to <br>
        para = para.replace("\n", "<br>")

        # Bold text **text**
        para = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", para)

        # Italic text *text* or _text_
        para = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", para)
        para = re.sub(r"_(.+?)_", r"<em>\1</em>", para)

        # Code `text`
        para = re.sub(
            r"`(.+?)`",
            r'<code style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 13px;">\1</code>',
            para,
        )

        # Convert URLs to clickable links
        para = re.sub(
            r"(https?://[^\s<]+)", r'<a href="\1" style="color: #2563eb; text-decoration: none;">\1</a>', para
        )

        # Greek letters
        para = para.replace("rho=", "\u03c1=")
        para = para.replace("ρ=", "\u03c1=")

        # Add paragraph
        html_parts.append(f'<p style="margin: 0 0 16px 0; color: #1a1a1a;">{para}</p>')

    return "".join(html_parts)


def create_plain_text(body: str, test_mode: bool = False, test_info: Optional[dict] = None) -> str:
    """Create plain text version of email."""
    lines = []

    if test_mode and test_info:
        lines.append("=== TEST EMAIL ===")
        lines.append(f"Original recipient: {test_info.get('recipient', 'unknown')}")
        lines.append(f"Draft file: {test_info.get('filename', 'unknown')}")
        lines.append("=" * 40)
        lines.append("")

    lines.append(body)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Best,")
    lines.append("Ivan Weiss Van Der Pol")
    lines.append("Jonathan Verdun")
    lines.append("Kyrian Weiss Van Der Pol")
    lines.append("")
    lines.append("AI Whisperers | CONACYT Paraguay")
    lines.append("https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics")

    return "\n".join(lines)


def create_html_email(body: str, test_mode: bool = False, test_info: Optional[dict] = None) -> str:
    """Create beautiful HTML version of email."""
    # Convert body to HTML
    body_html = markdown_to_html(body)

    # Add test banner if in test mode
    if test_mode and test_info:
        test_banner = f"""
        <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px; padding: 12px 16px; margin-bottom: 20px;">
          <p style="margin: 0; color: #92400e; font-size: 13px; font-weight: 600;">TEST EMAIL</p>
          <p style="margin: 4px 0 0 0; color: #92400e; font-size: 13px;">
            Original recipient: {html.escape(test_info.get("recipient", "unknown"))}<br>
            Draft file: {html.escape(test_info.get("filename", "unknown"))}
          </p>
        </div>
        """
        body_html = test_banner + body_html

    return HTML_TEMPLATE.format(body_html=body_html)


def parse_draft(filepath: Path) -> Optional[EmailDraft]:
    """Parse a markdown draft file into an EmailDraft object."""
    content = filepath.read_text(encoding="utf-8")

    # Extract status
    status_match = re.search(r"\*\*Status:\*\*\s*(.+)", content)
    status = status_match.group(1).strip() if status_match else "UNKNOWN"

    # Extract recipient email from **To:** line
    to_match = re.search(r"\*\*To:\*\*\s*(\S+@\S+)", content)
    if not to_match:
        print(f"  WARNING: No email found in {filepath.name}")
        return None
    recipient_email = to_match.group(1).strip()

    # Extract subject
    subject_match = re.search(r"\*\*Subject:\*\*\s*(.+)", content)
    if not subject_match:
        print(f"  WARNING: No subject found in {filepath.name}")
        return None
    subject = subject_match.group(1).strip()

    # Extract recipient name from filename or first heading
    name_match = re.search(r"# Email Draft: (.+)", content)
    recipient_name = name_match.group(1).strip() if name_match else filepath.stem

    # Extract email body - everything between "---" after Subject and before the next "---"
    # Find the EMAIL section
    email_section_match = re.search(
        r"## EMAIL\s*\n\n\*\*To:\*\*.+?\n\n\*\*Subject:\*\*.+?\n\n---\n\n(.+?)\n\n---", content, re.DOTALL
    )

    if not email_section_match:
        # Alternative pattern - look for body after subject line
        alt_match = re.search(r"\*\*Subject:\*\*.+?\n\n---\n\n(.+?)\n\n---\n\n## Notes", content, re.DOTALL)
        if alt_match:
            body = alt_match.group(1).strip()
        else:
            print(f"  WARNING: Could not extract body from {filepath.name}")
            return None
    else:
        body = email_section_match.group(1).strip()

    # Remove signature from body (we add it in template)
    # Find where signature starts
    sig_patterns = [
        r"\n\nBest,\n.*$",
        r"\n\nBest regards,\n.*$",
    ]
    for pattern in sig_patterns:
        body = re.sub(pattern, "", body, flags=re.DOTALL)

    # Count words in body
    word_count = len(body.split())

    return EmailDraft(
        filename=filepath.name,
        recipient_name=recipient_name,
        recipient_email=recipient_email,
        subject=subject,
        body=body,
        status=status,
        word_count=word_count,
    )


def get_all_drafts(drafts_dir: Path) -> list[EmailDraft]:
    """Get all parseable draft files."""
    drafts = []
    for filepath in sorted(drafts_dir.glob("[0-9]*.md")):
        draft = parse_draft(filepath)
        if draft:
            drafts.append(draft)
    return drafts


def send_email(
    draft: EmailDraft,
    sender_email: str,
    sender_password: str,
    test_mode: bool = True,
    test_email: Optional[str] = None,
    dry_run: bool = False,
) -> bool:
    """Send an email via Gmail SMTP with HTML formatting."""

    # Determine recipient
    actual_recipient = test_email if test_mode else draft.recipient_email

    if not actual_recipient:
        print("  ERROR: No recipient email available")
        return False

    if dry_run:
        print(f"  [DRY RUN] Would send to: {actual_recipient}")
        print(f"  Subject: {draft.subject}")
        print(f"  Body preview: {draft.body[:100]}...")
        return True

    # Create message with both HTML and plain text
    msg = MIMEMultipart("alternative")
    msg["Subject"] = draft.subject if not test_mode else f"[TEST] {draft.subject}"
    msg["From"] = f"AI Whisperers <{sender_email}>"
    msg["To"] = actual_recipient

    # Test info for headers
    test_info = {"recipient": draft.recipient_email, "filename": draft.filename} if test_mode else None

    # Attach plain text version first (fallback)
    plain_text = create_plain_text(draft.body, test_mode, test_info)
    msg.attach(MIMEText(plain_text, "plain", "utf-8"))

    # Attach HTML version (preferred)
    html_content = create_html_email(draft.body, test_mode, test_info)
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        # Connect to Gmail SMTP
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True
    except smtplib.SMTPAuthenticationError:
        print("  ERROR: Authentication failed. Check your App Password.")
        print("  Make sure you're using an App Password, not your regular password.")
        print("  Create one at: https://myaccount.google.com/apppasswords")
        return False
    except Exception as e:
        print(f"  ERROR sending email: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Send outreach emails from markdown drafts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--test", action="store_true", help="Test mode: send to test email instead of real recipients"
    )
    mode_group.add_argument("--real", action="store_true", help="Real mode: send to actual recipients (CAREFUL!)")
    mode_group.add_argument("--dry-run", action="store_true", help="Dry run: show what would be sent without sending")

    # Options
    parser.add_argument("--test-email", type=str, help="Email address for test mode (required with --test)")
    parser.add_argument("--draft", type=str, help="Send only this specific draft file (e.g., 01_brian_kuhlman.md)")
    parser.add_argument("--confirm", action="store_true", help="Required confirmation for --real mode")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between emails in seconds (default: 2.0)")
    parser.add_argument(
        "--status",
        type=str,
        default="READY TO SEND",
        help='Only send drafts with this status (default: "READY TO SEND")',
    )

    args = parser.parse_args()

    # Validate arguments
    if args.test and not args.test_email:
        parser.error("--test-email is required when using --test mode")

    if args.real and not args.confirm:
        parser.error("--confirm is required when using --real mode (safety check)")

    # Get credentials from environment
    sender_email = os.environ.get("GMAIL_ADDRESS")
    sender_password = os.environ.get("GMAIL_APP_PASSWORD")

    if not args.dry_run and (not sender_email or not sender_password):
        print("ERROR: Missing credentials. Set environment variables:")
        print("  export GMAIL_ADDRESS='your.email@gmail.com'")
        print("  export GMAIL_APP_PASSWORD='your-16-char-app-password'")
        print("\nTo create an App Password:")
        print("  1. Enable 2FA on your Google account")
        print("  2. Go to https://myaccount.google.com/apppasswords")
        print("  3. Create a new App Password for 'Mail'")
        sys.exit(1)

    # Find drafts directory
    script_dir = Path(__file__).parent
    drafts_dir = script_dir

    # Get drafts
    print(f"\nScanning drafts in: {drafts_dir}")
    all_drafts = get_all_drafts(drafts_dir)
    print(f"Found {len(all_drafts)} parseable drafts")

    # Filter by specific draft if provided
    if args.draft:
        all_drafts = [d for d in all_drafts if d.filename == args.draft]
        if not all_drafts:
            print(f"ERROR: Draft '{args.draft}' not found or not parseable")
            sys.exit(1)

    # Filter by status
    ready_drafts = [d for d in all_drafts if args.status in d.status]
    print(f"Drafts with status '{args.status}': {len(ready_drafts)}")

    if not ready_drafts:
        print("No drafts to send.")
        sys.exit(0)

    # Confirm before sending
    mode_str = "TEST" if args.test else ("DRY RUN" if args.dry_run else "REAL")
    print(f"\n{'=' * 50}")
    print(f"MODE: {mode_str}")
    print(f"FORMAT: HTML with plain text fallback")
    if args.test:
        print(f"All emails will be sent to: {args.test_email}")
    elif args.real:
        print("WARNING: Emails will be sent to REAL recipients!")
    print(f"Drafts to send: {len(ready_drafts)}")
    print(f"{'=' * 50}\n")

    if not args.dry_run and sys.stdin.isatty():
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    # Send emails
    success_count = 0
    fail_count = 0

    for i, draft in enumerate(ready_drafts, 1):
        print(f"\n[{i}/{len(ready_drafts)}] {draft.filename}")
        print(f"  To: {draft.recipient_name} <{draft.recipient_email}>")
        print(f"  Subject: {draft.subject}")
        print(f"  Words: {draft.word_count}")

        success = send_email(
            draft=draft,
            sender_email=sender_email or "test@example.com",
            sender_password=sender_password or "",
            test_mode=args.test,
            test_email=args.test_email,
            dry_run=args.dry_run,
        )

        if success:
            success_count += 1
            print("  [OK] Sent successfully")
        else:
            fail_count += 1
            print("  [FAIL] Failed to send")

        # Delay between emails (except for last one)
        if i < len(ready_drafts) and not args.dry_run:
            print(f"  Waiting {args.delay}s...")
            time.sleep(args.delay)

    # Summary
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total: {len(ready_drafts)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")

    if args.test:
        print(f"\nAll test emails sent to: {args.test_email}")
        print("Check your inbox (and spam folder)!")


if __name__ == "__main__":
    main()

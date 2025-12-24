# Testing Patterns

## 1. The Factory Pattern

Used for data generation. Centralizes the rules for creating valid entities.

- **Location**: `tests/factories/`
- **Usage**:
  ```python
  user = UserFactory.build(active=True)
  ```

## 2. The Page Object Model (POM)

Used for E2E testing to abstract the UI.

- **Location**: `tests/e2e_support/pages/`
- **Rule**: Tests never contain CSS/XPath selectors. Only Pages do.
- **Usage**:
  ```python
  login_page.enter_credentials(user.email, user.password)
  dashboard = login_page.submit()
  ```

## 3. The Builder Pattern

Used to construct complex scenarios.

- **Location**: `tests/core/builders/`
- **Usage**:
  ```python
  scenario = ScenarioBuilder()\
      .with_logged_in_user()\
      .with_product_in_cart()\
      .build()
  ```

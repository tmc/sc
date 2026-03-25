---
title: extensions.v1
description: API Specification for the extensions.v1 package.
---

<a name="common-proto"></a><p align="right"><a href="#top">Top</a></p>

<!-- begin services -->

<!-- begin services -->



<a name="extensions-v1-Author"></a>

### Author

Author provides attribution information.

IDENTITY:
At minimum, provide name OR email. For automated tools, name may be
the tool name (e.g., "zelda3-extractor").




| Field | Type | Description |
| ----- | ---- | ----------- |
| name |string| Display name.   |
| email |string| Email address.   |
| organization |string| Organization or team.   |
| identifier |string| Unique identifier (ORCID, GitHub username, etc.).   |
| role |string| Role in this context: "author", "reviewer", "approver".   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-License"></a>

### License

License captures licensing information.

SPDX COMPLIANCE:
Use SPDX identifiers for standard licenses:
  - "MIT", "Apache-2.0", "GPL-3.0-only"
  - "NOASSERTION" for unknown
  - "LicenseRef-<idstring>" for custom licenses

REFERENCES:
[SPDX] https://spdx.org/licenses/




| Field | Type | Description |
| ----- | ---- | ----------- |
| spdx_id |string| SPDX license identifier.   |
| name |string| Full license name.   |
| url |string| License URL.   |
| copyright |string| Copyright holder.   |
| year |string| Copyright year(s).   |
| notice |string| Additional notices or terms.   |
| detected |bool| Whether license was detected vs declared.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-Link"></a>

### Link

Link provides an external reference.

LINK TYPES:
Common types for categorization:
  - "documentation": User/developer docs
  - "design": Design documents, RFCs
  - "issue": Bug tracker, feature request
  - "figma": Figma design link
  - "confluence": Confluence wiki
  - "api": API documentation
  - "source": Source code reference




| Field | Type | Description |
| ----- | ---- | ----------- |
| title |string| Link title for display.   |
| url |string| URL (must be valid URI).   |
| type |string| Link type for categorization.   |
| description |string| Brief description of what the link contains.   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="extensions-v1-ChangeEntry"></a>

### ChangeEntry

ChangeEntry records a modification to an element.




| Field | Type | Description |
| ----- | ---- | ----------- |
| version |string| Version when change was made.   |
| timestamp |Timestamp| Timestamp of change.   |
| author |[Author](#extensions-v1-Author)| Author of the change.   |
| change_type |string| Type of change: "created", "modified", "reviewed", "approved".   |
| description |string| Description of the change.   |
| ticket |string| Issue/ticket reference.   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end messages -->

<!-- begin file-level enums -->
 <!-- end file-level enums -->

<!-- begin file-level extensions -->
 <!-- end file-level extensions -->


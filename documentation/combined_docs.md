\newpage

# What is AI App Store?

The AI App Store is a platform for accessing and operationalizing AI/ML applications and services that are built using [H2O Wave](https://wave.h2o.ai/docs/getting-started). The apps in the store are designed to help organizations incorporate AI into your business processes and make data-driven decisions.

## Why do I need it? 

The end goal of data science projects is to produce and host analytical software applications to facilitate decision-support and automated decision making.

The primary purpose of such applications is to help stakeholders make better decisions by giving them relevant information in an easily understandable format. Most of the heavy lifting is already taken care of by an app’s authors: what data to use, which algorithms to apply, what information to present, and how to present it.

Developing and deploying such applications presents some unique problems:

- **Infrastructure:** AI/ML modeling is storage and compute intensive. Incorporating machine learning into the software development process and integrating machine learning models into software applications is significantly more complicated compared to conventional software development.
- **Talent:** Building applications requires a cross-disciplinary team with specialized skills - data scientists, data engineers, backend/frontend engineers, and IT/operations - working in close collaboration with stakeholders.
- **Time to market:** Application requirements are rarely set in stone. Market conditions, competitor offerings, and customer expectations change all the time. Software development teams no longer have months or years to develop and deploy applications. There is an intense need to prototype quickly, gather early feedback from stakeholders, and improve iteratively or fail fast.

In other words, it requires extraordinary effort from a diverse team to wire up data, libraries, tooling and infrastructure before we can focus on what matters most: getting decision-support into the hands of stakeholders. This is where the H2O AI Cloud platform and the AI App Store comes in. 

## What can I do with it? 

H2O.ai’s AI App Store is a turnkey platform that streamlines this entire process: one platform and one API.

- **Turnkey infrastructure:** Provides all the building blocks and services necessary to develop and deploy analytical applications in one install. Combines data connectors, data storage, automatic machine learning, model operations, and rapid web application development into a single, scalable, vendor-neutral platform with a coherent, end-to-end API.
- **Empowers Python programmers:** The [Wave SDK](https://wave.h2o.ai/) makes it easy for data scientists and data engineers to develop beautiful, polished, low-latency, real-time analytical web applications using pure Python and publish them directly to end-users. No Javascript/HTML/CSS required.
- **Faster time to market:** Makes it easy to train models and immediately use them in interactive web applications for rapid prototyping and sharing with end-users. Dramatically simplifies and speeds up the iterative develop-deploy-feedback cycle.

\newpage

# Access AI App Store

Navigate to your organization's instance of [H2O AI Cloud](https://cloud.h2o.ai/login?referer=%2Fhome) and enter your login credentials. If you do not have an account or you are not sure which URL to access, contact your administrator or support team. 

![](appstore-login.png)


Once you are logged in, click **App Store** on the top navigation bar and you will see your AI App Store home page. 

![](appstore-homepage.png)

You can use the search bar to search for a particular app, or find the app you want by clicking on the relevant category listed on the left sidebar.

\newpage

# Architecture

The AI App Store is a core component of H2O AI Cloud (HAIC) and can only be used as a direct part of the HAIC platform. It is a turnkey platform that streamlines the process of developing, deploying, and using analytical ML software applications based on the H2O Wave development framework. The App Store server communicates with the Kubernetes API to schedule Wave apps, and it also authorizes and proxies all the traffic that comes through the Wave apps.

From an architectural perspective, App Store can be described as a replicated server that handles the following tasks:

* Handles the Wave app repository (that is, the list of available apps) and displays pictures and metadata related to apps.
* Instantiates those apps as running workloads on Kubernetes using scheduling. Uses [Helm](https://helm.sh/docs/) as an abstraction layer to communicate with the Kubernetes API.
* After apps are scheduled, they can be described as small containers that run Waved (the Wave server), the application Python code, and HAIC Launcher (a booster binary that ensures that the containers starts and operates correctly).
* Apps are accessed through the router component of the App Store server.
* The App Store server uses the Kubernetes API to store information about running apps. This means that HAIC is unable to distinguish between different methods used for manipulating apps (for example, if an app is started with Helm from the command line), which makes HAIC difficult to break even when different methods for manipulating apps are used. This applies to tasks like starting, updating, and deleting apps.
* The metadata database contains app metadata including locations of relevant icons, how should the app be started, and who owns the app.
* Metadata database (PostgreSQL):
    * Stores app metadata including tags and pointers to Blob Storage.
    * Doesn't store instance metadata.
    * Stores browser session data.
* It uses Blob Storage (S3/Azure Blob Storage/GCS) or Persistent Volume to store the app bundles (that is, `.wave` archives and extracted static app assets including icons and screenshots). Since Blob Storage allows for higher scalability and reliability, it is preferred over Persistent Volume whenever possible.
* Router:
    * Performs instance access authorization.
    * Routes requests to the relevant instance K8s service.
    * Consults scheduler to find the relevant K8s service.
* Scheduler:
    * Manages app instances through Helm client.
    * All instance metadata is stored in K8s API through Helm.
* Wave app instance:
    * 1-pod deployment with `clusterIP` service and optional PVC(s). The pod runs a single generic container with HAIC Launcher as the main process.
    * HAIC Launcher takes care of downloading the app code, installing its dependencies, starting Waved, and starting the app.


For more details about the architecture of the rest of the HAIC platform, see [HAIC architecture](https://docs.h2o.ai/haic-documentation/overview/architecture#app-store).

\newpage

# Concepts
AI App Store recognizes three actors:

* **App developer**: creates and publishes apps
* **App user**: browses and runs apps, can be either user with "full access" or visitor
* **Admin**: manages the platform

over five resource types:

* **App**: runnable Wave app package
* **App instance**: running instance of an app
* **App tag**: label for categorizing apps withing the platform
* **App secret**: sensitive information needed to run apps within the platform or dynamically injected configuration
* **App alias**: a custom URL for a particular app instance 

## App

App is a runnable Wave app package with metadata, such as (grouped into categories):

* Identity
  * a unique name and version identifier
* Display/search
  * a title and description
  * icon and screenshots
  * long description
  * tags/categories
* Authorization
  * owner (i.e., the person who imported it into AI App Store)
  * visibility (`PRIVATE`, `ALL_USERS`)
  * Instance lifecycle (`ON_DEMAND`, `MANAGED`)
* Runtime
  * RAM/disk requirements
  * other runtime settings (e.g., pointers to dependencies and secrets to be injected at startup time)

Users can start/run multiple instances of each app - subject to [Authorization](#authorization). 

Every authorized user can start their own instance. 

Apps are mostly **immutable**, meaning once uploaded, they cannot be changed, 
except for their App Store configuration options (see [Configuration options](#app-configuration-with-the-user-interface)).
To "update" an app, one has to upload a new version. This is to simplify the app lifecycle
and remove the need for developers to address app upgrade/downgrade.

See the [CLI](#apps) documentation for instructions on how to manage apps.

**Note:**
Internally, AI App Store treats every app name/version combination as a separate entity.
The UI then uses the app name to link several versions together; however each can have different
title, description, owner, instances, etc.




## App instance

App instance is a running instance of an app with the following metadata:

* pointer to the corresponding app
* owner (the person who started it)
* visibility (`PRIVATE`, `ALL_USERS`, `PUBLIC`)

The AI App Store fully manages the app instance lifecycle on behalf of its users.

Instances can be stateless or stateful (depending on the app configuration)
and can use external dependencies (e.g., AWS S3, Driverless AI).

Under the hood, each instance consists of several k8s resources, specifically, each instance is running in its
own k8s `pod`, under its own k8s `service`, accessible via a AI App Store subdomain (e.g., `https://1234.wave.h2o.ai`).
It can optionally include other resources, such as PVCs, Configmaps, etc.

See the [CLI](#app-instances) documentation for instructions on how to manage app instances.


## App tag

Tags are means of annotating apps in the platform (similar to
[GitHub issue labels](https://docs.github.com/en/free-pro-team@latest/github/managing-your-work-on-github/about-labels)).
Beyond visually categorizing apps, tags also act as a mechanism by which apps are exposed to "visitors" (i.e., users without "full access");
see [Authorization for visitors](#authorization-for-visitors) for details.

Tags are standalone resources with the following metadata (grouped into categories):

* Display/search properties
  * name, title, color, description
* ACLs
  * admin roles (i.e., the users that can manage the tag)
  * visitor roles  (i.e., the visitors that can view apps with this tag)
  
Tags are assigned to apps individually. Each tag can be assigned to multiple apps, and each app can have multiple tags assigned to it.

See the [CLI](#tags) documentation for instructions on how to manage tags.

## App secret

Secrets are backed by [Kubernetes secrets](https://kubernetes.io/docs/concepts/configuration/secret/) and a meant
for storing sensitive information that can be mounted as an environment variable or file.
Secrets are standalone resources with the following metadata:

* name
* visibility (`PRIVATE`, `ALL_USERS`, `APP`)
* parent (the parent scope of the secret; name + visibility + parent uniquely identify a secret)
* owner (the person who created it)

See the [CLI](#secrets) documentation for instructions on how to manage secrets.

## App alias

Aliases are essentially assignable custom URLs for app instances. 
By default, running instances of apps have URLs like `8184-810243981-23.cloud.h2o.ai`. 
Via an alias, we can expose the same instance under `my-awesome-app.cloud.h2o.ai`.

Aliases are standalone resources with lifecycles that are separate from app instances. They include the following metadata:

* name
* instance ID
* whether it is a primary alias or not

Having a separate lifecycle, an alias can be dynamically assigned to an instance or moved from
one instance to another.
If an instance corresponding to an alias is terminated, the alias will keep on existing but will become inactive,
returning HTTP `404` response for all attempts to visit it, until it is assigned to another instance.

One instance can have multiple aliases assigned, but each alias can only be assigned to one instance.

The alias marked as `primary` will serve as the actual URL for accessing the instance.
Accessing the instance via other aliases or via its UUID URL will result in a HTTP `302` redirect to the primary alias.
If an instance has no primary alias, then the UUID URL is considered to be the primary alias.
There can be, at most, one primary alias assigned to an instance.


See the [CLI](#aliases) documentation for instructions on how to manage aliases.

## Visibility

Visibility is a generic concept for governing some basic authorization rules for resources that do have this attribute, such as app, app instance, and app secret. For concrete rules and more information, see [Authorization](#authorization).

\newpage

# Using apps

## Finding an app on the App Store page

1. Go to the **App Store** page in [H2O AI Cloud](https://cloud.h2o.ai/) to find the app you want in
   a matter of seconds by looking through the categories or by simply typing part of the app's name
   or description in the search bar.

![](./assets/search-by-category.png) ![](./assets/search-in-searchbar.png)

2. Click the particular app tile to open the **App details** page.

   The app details page includes a description of the app, the app owner's details, the date and
   time app was created and last updated, and other metadata.

   It also shows your instances of the app and everyone else's
   instances [you have access to](#instance-authorization-for-users-with-full-access)
   .

## Running your own app instance

In the AI App Store each user is usually running their own instance(s) of an app as opposed to
sharing a single instance across all the users.

The App Store fully manages the app instance lifecycle on behalf of its users.
See [App instance](#app-instance) to find out more information.

Click the **Run** button to start your own instance of the app.

![](./assets/running-app-instance.png)

You can further manage the new instance on the [**My
Instances** page](#managing-instances-on-my-instances).

**Note:**
Not every app is runnable by every user. Some apps, like apps
with [managed instance lifecycle](#apps-managed-instance-lifecycle), are meant to
be shared by users and not started by each user individually.


## Visiting an instance

You can visit/use any app instance as long as the instance status is **deployed** and either you own
the instance or the visibility is set to **All Users**.

The best place to find and manage your own instances is on the [**My
Instances** page](#managing-instances-on-my-instances) or the [**App
Details** page](#finding-an-app-on-the-app-store-page) under the **My Instances** tab.

Instances owned by other people that you can visit/use can best be found via the [**App
Details** page](#finding-an-app-on-the-app-store-page) under the **All Instances** tab.

## Managing instances on My Instances

If the app instance is running, the status of the app instance shows up as **"Deployed"**. You can
visit your app instance only if its status is deployed.

Additionally, you can pause a running app instance by clicking the **Pause** button. Now the app
status will change to **"Paused"**.

You can click **Resume** to continue the execution of the app.

Also, you can terminate a deployed app instance by clicking **Terminate** from the drop-down menu.
The app instance will be deleted permanently.

![](./assets/managing-app-instance.png)

To see logs of a running instance, visit the **My Instances** page and click the **Instance log**
link in a particular app tile.

![](./assets/instance-log.png)

You can view the logs of the current process as well as the previous process. Also, to obtain the
entire log, you can simply click the **Download** button.

\newpage

# Authorization

Authorization rules differ depending on the role of a user, distinguishing between users with "full access",
visitors (users without "full access"), and admins.

## App authorization for users with full access

Access to apps is governed by the following rules:

* `ALL_USERS` apps are visible to all signed-in users with "full access"; they are also visible on the "App Store" page;
    these are typically created via `h2o bundle import`

* `ALL_USERS` apps with the `ON_DEMAND` instance lifecycle are runnable by all signed-in users with "full access"
* In all other cases the app owner is the only authorized user to perform a particular action, including:
  * `PRIVATE` apps are only visible to/runnable by the owner; these are only visible on the "My Apps" page and are typically experimental versions created via `h2o bundle deploy`
  * `ALL_USERS` apps with the `MANAGED` instance lifecycle are only runnable by the app owner.
  * The app owner can manage (view, run, update, delete, download) their apps via `h2o app ...` or via the "My Apps" page
* Any user with "full access" can import new apps into the platform via `h2o ...`
* `PUBLIC` apps are visible to all users; even if they are not logged in. However, the user must be logged in to use the app. 
  * Administrators must enable `config.publicModeEnabled`. If it is not enabled, authentication will be required and the behavior will be the same as `ALL_USERS` apps.

See [CLI](#apps) for details on managing apps.

## Instance authorization for users with full access

Access to app instances is governed by the following rules:

* `PRIVATE` instances are only visible to the owner (and to an extent to the owner of the corresponding app, see below for details)
* `ALL_USERS` instances are visible to all signed-in users with "full access"
* `PUBLIC` instances are visible to anyone on the Internet
* The instance owner can manage (view, update, terminate, see status/logs of) her instances via `h2o instance` or via the "My Instances" page
* App owner can see metadata, status, and logs of her app's instances via `h2o instance` or via the app detail page
  regardless of instance visibility; this is to facilitate troubleshooting;
  note that this does not include access to the app UI itself or any write access

Note that app/instance visibility can be modified by the owner, e.g., using `h2o (app|instance) update <id> -v <visibility>`
or via the "My Apps"/"My Instances" page.

See [CLI](#app-instances) for details on managing app instances.

## Tag authorization for users with full access

Access to tags is governed by the following rules:

* All users with "full access" can see all tags and tag assignments
* A tag can only be assigned/removed/updated by users having a role (as determined by the auth provider)
  that is present in the tag's `Admin Roles` list; empty means any user with "full access" is allowed

* Currently, tags can only be created by admins

See [CLI](#app-tags) for details on managing tags.

## Secret authorization for users with full access

Access to secrets is governed by the following rules:

* All users with "full access" can see all `ALL_USERS` secrets and their own `PRIVATE` secrets, but
  not secrets with visibility `APP` (see [App-scoped Secrets](#app-scoped-secrets))

* A `PRIVATE` secret can be created, updated, deleted by the user who created the secret
* Currently, `ALL_USERS` and `APP` secrets can only be created, updated or deleted by admins

See [CLI](#secrets) for details on managing secrets.

## Authorization for visitors

Visitors, a.k.a., users without "full access", have limited permissions within the platform:

* Visitors can only ever see their own instances, regardless of instance visibility (technically,
  they can also access UI of the `PUBLIC` instances, if given the URL)

* Visitors cannot see app logs, not even for their own instances  
* Visitors cannot import apps into the platform
* Visitors can only see/run `ALL_USERS` apps that have a tag which includes one of the visitor's roles
  (as determined by the auth provider) in the tag's `Visitor Roles`; empty means no visitors are allowed

  * *Example*: Visitor `UA` has role `RA`, visitor `UB` has role `RB`, tag `TA` has `Visitor Roles` `RA, RC`, tag
    `TB` has `Visitor Roles` `RB`, app `A1` has no tags, app `A2` has tag `TA`, app `A3` has tags `TA, TB` but is `PRIVATE`.
    In this case, user `UA` can see and run app `A2`, while `UB` cannot see or run any apps.

* Visitors cannot see tags or tag assignments
* Visitors cannot see secrets

## Authorization for admins

The admin API gives admins read/write access to all apps/instances/tags.
Note that the admin API does not allow access to the app UI itself, meaning admins cannot access UI of `PRIVATE` instances.
Similarly, admins cannot impersonate another user, e.g., for the purposes of importing/running an app.

\newpage

# Troubleshooting Guide

This guide helps you troubleshoot and understand any common issues or errors that you come across
while using the AI App Store.

### `failed scheduling app`

This error may occur when running/resuming an app. This error is an *internal error*, meaning the
App Store itself failed to fulfill the request due to circumstances outside its control or the
control of the user.

**Causes**

Typical causes for this error are related to Kubernetes or App Store configuration, such as:

- The Kubernetes cluster is out of capacity. In that case additional compute capacity must be added
  to schedule the app. When using autoscaling, it either hit the configured ceiling or scaling up
  the cluster took longer than the App Store timeout.

- The app refers to nonexistent secrets, which will prevent the app from starting, thus causing the
  App Store action to time out. App Store has validations that try to prevent this error, but it can
  still happen if the Kubernetes state is modified from the outside.

- The App Store runtime version or the server at large may be incorrectly configured, e.g. with an
  invalid GPU type, and as such the app cannot find a suitable node to run on.

- The App Store server and the Kubernetes cluster have been incorrectly configured w.r.t. taints and
  tolerations, such that the app cannot find a suitable node to run on.

- The Kubernetes control plane is just be too slow/overworked and has been unable to schedule the
  app in the alloted time.

- The container registry is temporarily unavailable/down, so the container image for the app cannot
  be pulled, or takes an excessive amount of time to pull.

**Mitigation**

Due to the nature of the error and for security reasons we do not report the details about the error
to the end user. So to resolve this error you may either try again in a little while or you have to
ask your administrator to consult the Kubernetes/App Store logs and determine the actual root cause
on your behalf.

---

<!---


### `error 2`

**Causes**

This error can occur due to one of the following reasons:

- reason 1
- reason 2

**Mitigation**

To resolve this, [insert steps on how to resolve]

---

### `error 3`

**Causes and Mitigation**

This error can occur due to one of the following reasons:

|                Possible Causes               |                                        Mitigation                                        |
|:--------------------------------------------:|:----------------------------------------------------------------------------------------:|
| Reason 1: explanation of possible root cause | How to resolve: steps on how to mitigate or fix the issue for this particular root cause |
| Reason 2: explanation of possible root cause | How to resolve: steps on how to mitigate or fix the issue for this particular root cause |

> **Notes for doc editors:**
    If each root cause has a different resolution, we could maybe use a table like the one seen above. This [markdown table generator](https://www.tablesgenerator.com/markdown_tables) was used to generate this table. Please delete this note and template sections when sending in your PR. 

-->

\newpage

# App developer guide

## App bundle structure

Each app must be bundled as a zip archive (commonly used with the suffix `.wave`  or `.zip`)
consisting of:

* `app.toml` - required; the platform configuration file
* `static/` - static asset directory, including the app icon (a png file starting with `icon`)
  and screenshots (files starting with `screenshot`)

* `requirements.txt` - pip-managed dependencies of the app (can contain references to `.whl` files
  included in the `.wave` using paths relative to the archive root)

* `packages.txt` - OS-managed dependencies of the app
* `.appignore` - specifies which files to ignore while bundling your app (the format is similar to `.gitignore` but with few [exceptions](https://github.com/monochromegane/go-gitignore#features))
* app source code

You can quickly create a `.wave` archive by running `h2o bundle` in your app git repository
(see the [CLI](#publishing-an-app-for-others-to-see-and-launch) section).

**Note:**
H2O AI Cloud supports the following runtimes:

- Python 3.8  | CPU | deb11_py38_wlatest
- Python 3.9  | CPU | deb11_py39_wlatest
- Python 3.10 | CPU | deb11_py310_wlatest
- Python 3.8  | GPU | ub2004_cuda114_cudnn8_py38_wlatest
- Python 3.10 | GPU | ub2204_cuda121_cudnn8_py310_wlatest



### app.toml

Each app archive has to contain an `app.toml` configuration file in the [TOML](https://toml.io/en/)
format, placed in the root of the `.wave` archive, example:

```toml
[App]
Name = "ai.h2o.wave.my-app"
Version = "0.0.1"
Title = "My awesome app"
Description = "This is my awesome app"
LongDescription = "LongDescription.md"
Tags = ["DATA_SCIENCE"]
InstanceLifecycle = "ON_DEMAND" 
InstanceTimeout = "1h"

[Runtime]
Module = "app.run"
VolumeMount = "/data"
VolumeSize = "1Gi"
ResourceVolumeSize = "2Gi"
MemoryLimit = "500Mi"
MemoryReservation = "400Mi"
CPULimit = "1.5"
CPUReservation = "500m"
GPUCount = 1
GPUType = ""
EnableOIDC = false
EnableSHM = false
RoutingMode = "DOMAIN"

[[Env]]
Name = "ENVIRONMENT_VARIABLE_NAME"
Secret = "SecretName"
SecretKey = "SecretKeyName"

[[Env]]
Name = "ANOTHER_ENVIRONMENT_VARIABLE_NAME"
Value = "some value"

[[File]]
Path = "some/path.file"
Secret = "FileSecretName"
SecretKey = "FileSecretKeyName"

[[File]]
Path = "/another/path.file"
Value = '''
some
string
'''
```

### Required attributes

**`[App]`**

| Attribute |  Type  | Description                                                                             |
|:---------:|:------:|-----------------------------------------------------------------------------------------|
| Name      | string | A machine-oriented unique identifier of the app (links different app versions together) |
| Version   | string | A [semver](https://semver.org) version of the app                                       |

**Note:**

`{{App.Name}}` and `{{App.Version}}` must be 63 chars or less and match the regex `^([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]$`


**`[Runtime]`**

| Attribute |  Type  | Description                                                                                                                                                                                                                                                |
|:---------:|:------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Module    | string | The name of the main Python module of the app, that is, the app should be started via `python3 -m $module_name` (this is only required if the value of the `App.InstanceLifecycle` attribute is not `LINK`; see [Link apps](#link-apps) for more information)  |

**Note:**

`{{App.Name}}-{{App.Version}}` must be 63 chars or less and match the regex `^([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]$`



### Optional attributes

**`[App]`**

|     Attribute     |       Type      | Description                                                                                                                                                                                                                                                                                |
|:-----------------:|:---------------:|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Title             | string          | A human-oriented name of the app for presentation in UI/CLI                                                                                                                                                                                                                                |
| Description       | string          | A single-line description of the app for presentation in UI/CLI                                                                                                                                                                                                                            |
| LongDescription   | string          | A path to a file, relative to the archive root, containing additional multi-line markdown description of the app. Although there is no actual restriction of the Markdown format, it is recommended to limit it to bullet-point lists (`*`), H3 headings (`###`), and hyperlinks (`[]()`). |
| Tags              | list of strings | Tags to automatically assign to the app upon importing. Apps can be identified by tag name. If the tag is listed as a category tag in the server configuration, the app will be assigned that category upon import.                                                                        |
| InstanceLifecycle | string          | Identifies the instance lifecycle, which can be set to [`ON_DEMAND`] or [`MANAGED`]. This config defaults to `ON_DEMAND` when empty. |
| InstanceTimeout   | string          | Overrides server defaults for auto pausing an app instance ("s", "m", "h")  are valid units, -1 for no timeout. Ex. "1h" |

**`[Runtime]`**

| Attribute          |  Type  | Description                                                                                                                                                                                                                                                                                                                                                                               |
|--------------------|:------:|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RuntimeVersion     | string | The name of the runtime version that the app will run on top of (similar to a docker base image, see [Runtime environment](#runtime-environment)). This config defaults to the platform's default when left empty/unspecified. Valid values differ based on the platform deployment and configuration.                                                                                    |
| VolumeMount        | string | The absolute path of the volume used to persist app instance data across restarts                                                                                                                                                                                                                                                                                                         |
| VolumeSize         | string | The volume size. This config value must conform to the [k8s resource model](#resource-quantities)                                                                                                                                                                                      |
| ResourceVolumeSize | string | The volume used to persist internal app resources (such as Python venv) across restarts. This is only recommended for production-quality apps with sizeable resources, due to cluster node limits. This config value must conform to the [k8s resource model](#resource-quantities).   |
| MemoryLimit        | string | A hard limit on the maximum amount of memory an instance can use before it is OOM-killed. This config defaults to service-wide settings managed by Admins (it is recommended to be conservative with these limits) and must conform to the [k8s resource model](#resource-quantities). |
| MemoryReservation  | string | The amount of memory required to schedule an instance of the app. This config defaults to service-wide settings managed by Admins (it is recommended to be conservative with these limits) and must conform to the [k8s resource model](#resource-quantities).                           |
| CPULimit           | string | Maximum CPU usage that an instance of the app can use. This config defaults to service-wide settings managed by Admins (it is recommended to be conservative with these limits) and must conform to the [k8s resource model](#resource-quantities).                                    |
| CPUReservation     | string | The number of CPU units required to schedule an instance of the app. This config defaults to service-wide settings managed by Admins (it is recommended to be conservative with these limits) and must conform to the [k8s resource model](#resource-quantities).                       |
| GPUCount           | int    | The number of GPU units required to schedule an instance of the app                                                                                                                                                                                                                                                                                                                        |
| GPUType            | string | GPU type required for the app. This config defaults to the platform's default when left empty/unspecified. Valid values differ based on the platform deployment and configuration.                                                                                                                                                                                                        |
| EnableOIDC         | bool   | Enable Wave to be set up with OIDC authorization, thereby giving access to the user's authentication and authorization information from your app (see [Wave security](https://wave.h2o.ai/docs/security/#single-sign-on) for details).                                                                                                                                                    |
| EnableSHM          | bool   | Enable extended docker shared memory for the app; some Libraries like `pyTorch` use [Shared Memory](https://en.wikipedia.org/wiki/Shared_memory) for Multiprocessing (see [this Kubernetes issue](https://github.com/kubernetes/kubernetes/issues/28272) for more details on this topic.                                                                                                  |
| RoutingMode        | string | The routing mode to be used for instances of the app can be set to either `DOMAIN` or, `BASE_URL`. This config defaults to `DOMAIN` when empty (see [App routing mode](#app-routing-mode) for details).
| AppMode            |   string  | App mode can be set to `python` or `container`. This config defaults to `wave` when empty.   |
| Port               |    int    | Port number of the app only valid when AppMode is python or container. |
| CustomImage        |   string  | If AppMode is container set the a custom container for this App  and must be imported using h2o admin app import |
| AppArgs            | \[]string | If AppMode is container set arguments to start the container. |
| AppEntryPoint      | \[]string | If AppMode is container set the entrypoint for the container. |

**`[Env]`** 

This struct contains configs that request for a literal value/secret to be injected into an instance at startup-time as an Env variable (see [Utilizing secrets](#utilize-app-secrets) for more details).

| Attribute |  Type  | Description                                                                                                                                                                                                           |
|-----------|:------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Name      | string | The name of the Env variable to the injected into the Python process. Names prefixed with `H2O_CLOUD` or prefixed with `H2O_WAVE` are disallowed (except `H2O_WAVE_APP_MODE` and names allowed by the administrator).  See [Configuring your app](https://wave.h2o.ai/docs/configuration#configuring-your-app) for a full list of environment variables that you can configure. |
| Secret    | string | The name of the shared secret to use. Each secret can contain multiple key-value pairs but cannot be used together with the `Value` config.                                                                           |
| Optional    | bool | If set to true the secret will not be required to exist to be imported. This config cannot be used together with `Value` config.                                                                                      |
| SecretKey | string | The name of the key within the secret that is to be used. This config cannot be used together with the `Value` config.                                                                                                |
| Value     | string | The literal value of the Env variable. This config cannot be used together with the `Secret` or `SecretKey` configs.                                                                                                  |



**`[File]`** 

This struct contains configs that request for a literal value/secret to be injected into an instance at startup-time as a file (see [Utilizing secrets](#utilize-app-secrets) for more details).

| Attribute |  Type  | Description                                                                                                                                                                                                                                                                                                    |
|-----------|:------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Path      | string | The path to inject the file into. A relative path is relative to the directory with the app code as determined by the platform. The path dir cannot be `/` or `.`  (only subdirs are allowed) and it must be unique across all other `File` configurations. Note that the `/resources` path dir is disallowed. |
| Secret    | string | The name of the shared secret to use. Each secret can contain multiple key-value pairs but cannot be used together with the `Value` config.                                                                                                                                                                    |
| Optional    | bool | If set to true the secret will not be required to exist to be imported. This config cannot be used together with `Value` config.                                       |
| SecretKey | string | The name of the key within the secret that is to be used. This config cannot be used together with the `Value` config.                                                                                                                                                                                         |
| Value     | string | The literal value of the Env variable. This config cannot be used together with the `Secret` or `SecretKey` configs.                                                                                                                                                                                           |


**`[Link]`** 

This struct is to be filled only if the value of `App.InstanceLifecycle` is `LINK` (see [Link apps](#link-apps) for details).

| Attribute |  Type  | Description               |
|-----------|:------:|---------------------------|
| Address   | string | Full URL of the app link. |


## Runtime environment

The platform executes each app in an environment given by its `RuntimeVersion`.
The `RuntimeVersion` determines the OS, Python version, location of the app code/venv, etc.

Developers can also specify the pip-managed dependencies of the app via standard `requirements.txt` (can contain
references to `.whl` files included in the `.wave` using paths relative to the archive root)

Developers can also specify the OS-managed dependencies of the app via `packages.txt` in a format
similar to `requirements.txt`: one package name per line.
These will be installed as given using the package manager given by the `RuntimeVersion`
(e.g., `apt-get install`).

Developers can further customize the runtime environment by [Utilizing secrets](#utilize-app-secrets).

The `h2o env runtime-version list` command will list the runtime-versions available for use.

```sh
$ h2o env runtime-version list
NAME                              	STATUS
deb11_py310_wlatest                	Default
ub2004_cuda114_cudnn8_py38_wlatest
...
```

**Note:**

The platform does not currently provide any provisions for developers to customize the OS and
Python versions beyond choosing a specific `RuntimeVersion`.

We are actively working on improving this.



### Standard environment variables

When running in an actual App Store deployment, each app instance will be configured with several 
standard environment variables specifying its context 
(these might be empty when running locally, e.g., via `h2o exec`), including:

* `H2O_CLOUD_ENVIRONMENT` - typically URL of the App Store deployment
* `H2O_CLOUD_INSTANCE_ID` - ID of the app instance
* `H2O_CLOUD_APP_NAME` - name of the app
* `H2O_CLOUD_APP_VERSION` - version of the app
* `H2O_CLOUD_APP_ID` - ID of the app

## How-To

### Update an app to a newer version

The "Catalog" page shows apps with visibility `ALL_USERS`, so rolling out a new app version is done by:

1. [importing a new version](#publishing-an-app-for-others-to-see-and-launch) of the app as `PRIVATE`
1. testing the new version
1. [changing the visibility](#updating-app-visibility) of the new version to `ALL USERS`
1. (optional) [changing the visibility](#updating-app-visibility) of the old version to `PRIVATE`

This is based on the [Basic concepts](#app):

> Apps are mostly **immutable**, meaning once uploaded, they cannot be changed (except for visibility).
> To "update" an app, one has to upload a new version.

and:

> Internally, AI App Store treats every app name/version combination as a separate entity.
> The UI then uses the app name to link several versions together; however each can have different
> title, description, owner, instances, etc.

An important corollary is that **instances of the old app version are not affected by the update**
(as they are completely separate from the new app version). The update only prevents users from
starting new instances of the old version.

### Pause or restart an app instance

The `h2o instance suspend <instanceId>` command pauses a running instance of a particular app. The app status changes to "Paused". You can configure `ResourceVolumeSize` in the [app.toml file](#optional-attributes) to utilize [Wave checkpointing](https://wave.h2o.ai/docs/configuration#h2o_wave_checkpoint_dir).

```sh
$ h2o instance suspend 2efe9ed7-2bdd-4d02-9be6-f73a196d663d
ID           	2efe9ed7-2bdd-4d02-9be6-f73a196d663d                     	
App          	                                                         	
App ID       	492dbac1-3230-413e-852f-11cb82b57436                     	
Created At   	2022-09-16 08:28:04                                      	
Updated At   	2022-12-12 09:03:23                                      	
Status       	SUSPENDED                                                	
Visibility   	ALL_USERS                                                	
URL          	https://2efe9ed7-2bdd-4d02-9be6-f73a196d663d.cloud.h2o.ai	
Owner        	oshini.nugapitiya@h2o.ai                                 	
Readonly     	false                                                    	
Suspendable  	true                                                     	
Suspend After	    
```
The `h2o instance resume <instanceId>` command restarts a paused or suspended instance of a particular app. The app status changes to "Deployed".  Any files that are saved to disk are not available after the restart unless they are in the directory in `VolumeMount`.

**Note:**
`VolumeMount` cannot be an existing folder in the App Bundle.


**Note:**
There is no guarantee that the Wave `on_shutdown` hook will be given time to complete when an app is suspended, the underlying Kubernetes pod moves nodes, or a node experiences a failure.


```sh
$ h2o instance resume 2efe9ed7-2bdd-4d02-9be6-f73a196d663d
ID           	2efe9ed7-2bdd-4d02-9be6-f73a196d663d                     	
App          	                                                         	
App ID       	492dbac1-3230-413e-852f-11cb82b57436                     	
Created At   	2022-09-16 08:28:04                                      	
Updated At   	2022-12-12 08:56:32                                      	
Status       	DEPLOYED                                                 	
Visibility   	ALL_USERS                                                	
URL          	https://2efe9ed7-2bdd-4d02-9be6-f73a196d663d.cloud.h2o.ai	
Owner        	oshini.nugapitiya@h2o.ai                                 	
Readonly     	false                                                    	
Suspendable  	true                                                     	
Suspend After	2022-12-12 16:56:31  
```

### Utilize app secrets

Developers can pass secrets registered with the platform to apps, exposed as environment variables
using the `[[Env]]` section within the `app.toml` or as files using the ``[[File]]`` section. Each
specific value from the secret that you want to use in your app needs its own environmental variable
or file path.

```toml
[[Env]]
Name = "USER_ENV_VAR"
Secret = "private-secret"
SecretKey = "user"

[[File]]
Path = "some/password.file"
Secret = "private-secret"
SecretKey = "password"
```

```python
import os


@app('/')
async def serve(q: Q):
    environment_variable = 'USER_ENV_VAR'
    default_value = 'user'
    value = os.getenv(environment_variable, default_value)
    q.page['my_card'] = ui.form_card(box='1 1 4 4', items=[ui.text(value)])
    await q.page.save()
```

This lets developers parametrize their apps with links to external dependencies (e.g., S3,
Driverless AI) securely, while allowing easy overrides for local development or deployments outside
the platform.

See [CLI documentation](#secrets) for instructions on manipulating secrets.

**Note:**

Apps imported into the App Store [directly](#publishing-an-app-for-others-to-see-and-launch)
can reference only `PRIVATE` secrets of the same user or `ALL_USERS` secrets.

`APP` secrets are reserved for apps imported via
the [H2O marketplace](#app-secrets).



### App routing mode

The app routing mode (`Runtime.RoutingMode` in [app.toml](#apptoml)) determines how the app
instances' UI is exposed to end users. The currently supported values are

* `DOMAIN` - each app instance is exposed on its own subdomain of the Appstore server,
  i.e., `uuid.appstore.domain/app/path...`. This is the default setting.

* `BASE_URL` - each app instance is exposed on a sub-path of the appstore server, i.e.,
  `appstore.domain/instance/uuid/app/path...`. This setting requires that the app itself supports
  serving behind a base URL/path. All apps
  using [Wave SDK 0.20](https://github.com/h2oai/wave/releases/tag/v0.20.0) and later should support
  this out of the box. The `/app/path...` portion of the URL path is forwarded to the app container
  via the `H2O_WAVE_BASE_URL` environment variable in case it is needed by your application for
  some reason (in most cases, however, the Wave SDK should handle this for you).

In both cases the app's webserver gets the full request URI as received by the Appstore server.

**Redirects:** The Appstore server's app router component implements a redirect (via 307
TemporaryRedirect HTTP status) in case an instance is accessed via the wrong URL, i.e. it redirects 
from sub-path to subdomain for instances with `DOMAIN` `RoutingMode` and vice versa for `BASE_URL`.

### App route

While it is not a strict requirement, since the platform deploys each app with its own Wave server,
we advise that apps use `/` as their main route:

```python
@app('/')
async def serve(q: Q):
    pass
```

### Give an app instance a pretty URL

You can leverage [App aliases](#app-alias) to give your instances 
pretty URLs, so that instead of `8184-810243981-23.cloud.h2o.ai` your users can access the instance 
via something like `my-pretty-name .cloud.h2o.ai`.

**Prerequisite:** You must be an admin of the environment to run these commands.

To create a single alias for an app for which you want to have a pretty URL, run:

```sh
$ h2o admin alias create my-pretty-name <instance id> True
```

This instance then can be accessed via `my-pretty-name.cloud.h2o.ai`, 
accesses to `<instance id>.cloud.h2o.ai` will result in an HTTP `302` redirect to the alias.

When you’ve created a new app instance, usually because there’s a new version of the app, you may
want to change which instance the alias points to. To do this, run:

```sh
h2o admin alias assign my-pretty-name <new instance id>`
h2o admin alias promote my-pretty-name
```

Note that there can be a slight delay before the change gets propagated.

See the [CLI](#aliases) documentation for details on these commands.

**Note:**

Please note that if the environment requires [base URL app routing](#app-routing-mode) for all apps,
you will need to add this alias to the App Store TLS certificate.



### Link apps

The App Store allows importing apps that do not bundle executable code (and thus cannot have
instances) but only represent a link to an external website. 
This kind of app is referred to as a "Link App".
The goal is to inject an external link into the App Store UI/CLI in a way that is
consistent in terms of UX with regular apps (app icon, listing on App Store page, categorization,
app details with long description and screenshots, etc.).

You can create a link app by setting `LINK` as the value of `App.InstanceLifecycle` in `app.toml`.
In such a case, you also need to set the `Link.Address` value to a URL of your choice.
The UI and CLI will then direct users to this URL instead of directing them to an instance of the app.

A link app bundle still contains `app.toml` and `static/`, but should not contain any source code or
`requirements.txt`.

A link app can leverage all the parameters in the `App` section, however the `Runtime` and `File` 
sections must be empty. For example:

```toml
[App]
Name = "ai.h2o.examples.link"
Version = "0.0.1"
Title = "Example: Link"
Description = "Showcasing the link functionality."
InstanceLifecycle = "LINK"

[Link]
Address = "https://h2o.ai" 
```

### Container Apps

The App Store allows importing containers in addition to Wave apps by admin users only.

You can create a container app by setting `container` as the value of `Runtime.AppMode` in `app.toml`.
In such a case, you can also specify `Runtime.Port` value to the port where the web application is running on. You can also specify `Runtime.AppArgs` and `Runtime.AppEntryPoint` to specify the entrypoint and arguments to start the container otherwise the default entrypoint or the container will be used. `Runtime.CustomImage` can be set to specify where the container is.

A container app bundle still contains `app.toml` and `static/`, but should not contain any source code or
`requirements.txt`.

In order for containerized apps to be imported the administrator must configure the server config option `config.allowedCustomImageRegexes` with the allowed registries containers can be imported from.

```yaml
  allowedCustomImageRegexes:

    - "^123456\\.dkr\\.ecr\\.us-east-1\\.amazonaws.com\\/.*"
```


```toml
[App]
Name = "ai.h2o.examples.container"
Version = "0.0.1"
Title = "Example: Container"
Description = "Showcasing the container functionality."

[Runtime]
AppMode = "container"
CustomImage = "customImage:latest"
AppArgs = ["--mode", "production"]
AppEntryPoint = ["./server"]
Port = 1337
```

To import this container app it must be bundled `h2o app bundle` and then imported using
`h2o admin app import` with the optional `--set-image` flag where the container image address
can be overridden.

In the case a container app is imported without an image it will be not runnable. The container image can be set using `h2o admin app set-image <appID> <image>` 
and then you must run `h2o admin app preconditions refresh <appID>` or wait until the appstore refreshes all the apps periodically to check if they are runnable.

### Configure an app via the user interface

While tags and visibility for an App can be configured via the CLI, these attributes can also be set using the user interface, as described below:

**Note:**

Since the user interface is being continually improved, it is possible that the images below will not match exactly what you see in your version of H2O AI Cloud.



On the "My Apps" page, click on the pencil icon for the app you wish to edit:

<img src={require('./developer-guide-assets/edit_app.png').default} 
     alt="My Apps - edit icon" width="300" 
     style={{boxShadow: '5px 5px 10px rgba(0,0,0,0.35)', marginBottom: 25}} />

This will bring up a panel on the right side of the user interface which will allow you to edit the values for:

* Visibility (See [Visibility](#visibility) for more information)
* Categories
* Badges
* Authorization Tags

<img src={require('./developer-guide-assets/app_config.png').default} 
     alt="App configuration panel" width="500" 
     style={{boxShadow: '5px 5px 10px rgba(0,0,0,0.35)', marginBottom: 25}} />

Tags will show up in the "Categories", "Badges", or "Authorization Tags" select menus based on the following criteria:

* If the tag has `isCategory` set to `true` it will be treated as a "Category", which allows it to be filtered on the App Store's main page.
* If the tag has one or more `visitorRoles` set, it will be treated as an "Authorization Tag".
* Otherwise, the tag will serve as a "Badge" in the App Store UI: 
  <img src={require('./developer-guide-assets/my_tag.png').default} alt="Example badge" width="100" align="top"/>
  
  The badge tags let the developer or the system administrator share more information about the app with end users. 
  For instance, the administrator can configure an open-source badge if your environment has many open-source applications. 
  
  Then the developers can tag their open source apps with this badge if they want to indicate to the user that the code is available in GitHub. This additional information shows on the App details page, the My apps page, and the Admin apps page.

  <img src={require('./developer-guide-assets/open-source-badge-tag.png').default} alt="badge tag" width="100%" align="top"/>

See [App Tag](#app-tag) to learn more about tags.

\newpage

# Using the H2O command line interface

The `h2o` command line interface (CLI) is a command line tool for app developers and other users
to interact with the platform.

<a href="https://h2oai-cloud-release.s3.amazonaws.com/releases/ai/h2o/h2o-cli/latest/index.html">
<button style={{backgroundColor: '#FECB2F', color: '#000000', padding: '15px 25px', fontSize: '18px', border: 'none', borderRadius: '50px'}}>Download latest CLI</button>
</a>

## Configuring the CLI

The following steps describe how to install and configure the `h2o` CLI to talk to a particular platform deployment.

1. [Download the latest CLI](https://h2oai-cloud-release.s3.amazonaws.com/releases/ai/h2o/h2o-cli/latest/index.html).

2. Add the `h2o` CLI binary to your $PATH. If you don't know how to do this, contact your support team.

3. On macOS and Linux machines, run `chmod +x h2o` from the terminal to give the CLI permission to run on your machine. Note that if you are using a Windows machine, you can skip this step.

4. Configure your CLI by running `h2o config setup`. You are prompted to fill in values for the following fields (note that the provided values are for demonstrative purposes only):

	```
	Endpoint URL: "https://h2o-cloud.example.com"
	OIDC URL: "https://auth.example.com"
	Client ID: "h2o-cloud-public"
	```

 To find the specific values to enter for the preceding fields according to your environment, you can click your username on the top-right corner of the H2O AI Cloud home screen, and then click **CLI & API access**.

 ![](cli/cli-api-access.png)

 You will see the following screen with the generated values for `Endpoint URL`, `OIDC URL`, and `Client ID`.

 ![](cli/cli-generated-values.png)

 Afterwards, you will be asked to visit the Endpoint URL to retrieve a token and paste that in to complete the configuration.

### Getting a new CLI token

You can get a new token without needing to go through the steps of creating an entirely new configuration.
This can be done like this: `h2o config update-token`.

### Using multiple config files

- You can have as many config files as you wish.

- When you run `h2o config setup`, your config will be saved at `~/.h2oai/h2o-cli-config.toml`. 

- You can have more than one config locally, so you can easily run commands against different environments. 

  > For example, you could have both `h2o-cli-config.toml` as the default which points to your production environment, and then another one called `h2o-cli-config-staging.toml` which references a different cloud instance. 

- When using the CLI with an alternate config other than the default, start all of your commands with `h2o --conf path/to/config.toml` or define the environment variable `H2OCONFIG` to let the CLI know which configuration to use. 

  > For example, when bundling an app to deploy to a different environment, you can run `h2o --conf ~/.h2oai/h2o-cli-config-staging.toml bundle import` or `H2OCONFIG=~/.h2oai/h2o-cli-config-staging.toml h2o bundle import`. When both the `H2OCONFIG` environment variable and `--conf` arguments are provided, the `--conf` argument take precedence.


## Platform token

The `h2o platform [token-info, refresh-token, access-token]` commands let you access the **H2O AI
Cloud platform token**.

The platform token allows you to securely authenticate against all the APIs in the H2O AI Cloud,
such as Enterprise Steam, Driverless AI, MLOps, AI App Store, Document AI, etc.
The platform token is an OpenID Connect (OIDC) refresh token that is used later to obtain API access
tokens, which are needed to use these APIs. This lets you securely orchestrate API calls from your
local workstation, Jupyter notebooks, or basically anywhere else.

For more details,
see [API authentication](https://docs.h2o.ai/haic-documentation/guide/general/using-platform-token#using-the-platform-token).

**Note:**
The platform token must be enabled in the environment for the following steps to work. If
you have issues replicating the following steps, contact your administrator to enable it.


You can obtain a new platform refresh token by following these steps.

```
$ h2o config update-platform-token
Visit https://your.domain.com/auth/get-platform-token to generate a new token and paste it below:
Token: <REDACTED>
```

The platform token is then cached in the CLI config file and can be printed via
`h2o platform token-info` (useful in scripts,
see [Using the platform token](https://h2oai.github.io/haic-documentation/guide/general/using-platform-token#using-the-platform-token))
and `h2o platform refresh-token` or converted to a fresh, short-lived access token
via `h2o platform access-token` for direct use in requests, such as via `curl`.

The cached platform refresh token can be updated any time via `h2o config update-platform-token`,
and it needs to be explicitly refreshed when it expires or is invalidated by the admin (depending on
environment configuration).

In addition to having a valid access token for API requests, you need the following information to connect to each component API:

```
MLOps: <value of the h2oai-mlops gateway Secret Key>
Steam: <value of the h2oai-steam API Secret Key>
```

The `h2o secret list` command will list all the secrets to which the user has access.

To list the secrets with the visibility type `ALL_USERS`, use the command `h2o secret list -v ALL_USERS`.

```sh
$ h2o secret list
NAME                 	VISIBILITY	PARENT	KEYS
h2oai-mlops          	ALL_USERS 	      	gateway
h2oai-steam          	ALL_USERS 	      	api, public-address
```

To obtain the value of the `h2oai-mlops` gateway, run `h2o secret get h2oai-mlops -e`.

To obtain the value of the `h2oai-steam` API, run `h2o secret get h2oai-steam -e`.

## Apps

The `h2o app [get, list, import, delete, run, update, meta]` commands let you see and, when
authorized, manage or run available apps.

### Listing existing apps

The `h2o app list -a` command will list all apps visible to the user.

```sh
$ h2o app list -a
ID                                    TITLE                        OWNER            CREATED VISIBILITY    TAGS
abc543210-0000-0000-0000-1234567890ab Peak 0.1.1                   user1@h2o.ai     18d     ALL_USERS     Beta
bcd543210-1111-1111-1111-0123456789ab Tour 0.0.15-20200922162859   user2@h2o.ai     20d     ALL_USERS
...
```

### Launching existing apps

To launch an app, the `h2o app run <appId>` command can be used to launch a new instance of that app.
The `-v` flag can be used with `app run` to specify app instance visibility settings.

```sh
$ h2o app run bcd543210-1111-1111-1111-0123456789ab
ID  22222222-3333-4444-5555-666666666666
URL https://22222222-3333-4444-5555-666666666666.cloud.h2o.ai
```

### Retrieving metadata from an app

The `h2o app meta <appId>` command can be used to retrieve `requirements.txt`, `packages.txt`, `app.toml` and a list of files from an app.
The `-t` flag can be used with `app meta` to specify `REQUIREMENTS`,`PACKAGES`, `FILES`, `APP`.

```sh
$ h2o app meta  19b2cc66-e1c3-4cfa-96eb-b00cdc8c0da0
h2o-wave==0.16.0
# Packages
$ h2o app meta 19b2cc66-e1c3-4cfa-96eb-b00cdc8c0da0 -t PACKAGES
java
libavcodec58
# Files
$ h2o app meta 1ed9a149-e6ab-41db-ab4a-a64630ad333a -t FILES
/
LoremIpsum.md
app.toml
lorem.py
requirements.txt
static/
static/icon-example.png
# app.toml
$ h2o app meta f1cea5b0-dfeb-46b2-b538-37d2605cf638 -t APP

[App]
Name = "lorem-ipsum"
Version = "0.0.1"
Title = "Lorem Ipsum"
Description = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque auctor."
LongDescription = "LoremIpsum.md"
SetupGuide = "setup_guide.txt"

[Runtime]
Module = "lorem"
```

### Inspecting a local app bundle

The `h2o app inspect <path>` command can be used to retrieve `requirements.txt`, `packages.txt`, `app.toml` and a list of files  from an app bundle
The `-t` flag can be used with `app meta` to specify `REQUIREMENTS`,`PACKAGES`, `FILES`, `APP`.

```sh
h2o app inspect ai.h2o.test.0.0.1.wave
App Toml    	[App]
              Description = "Test App "
              Name = "ai.h2o.test"
              Title = "Test Title"
              Version = "0.0.1"

              [Runtime]
              Module = "main"

Packages    	(no Packages)
Requirements	h2o-wave

Files       	Name
              /
              README.md
              app.toml
              main.py
              requirements.txt
              static/
              static/icon_model_analyzer.png
```

### Airgapped bundles

**Note:** ALPHA feature
Airgapped bundle creating is currently in ALPHA stage, breaking changes to the CLI interface are expected.


The `h2o bundle ...` command can be used to construct airgapped bundles. The bundle consists of
three components:

1. Dockerfile

    - used to create docker container containing the app and all its dependencies
2. Helm charts

    - used to import the app into the App Store during deployment of HAIC
    - currently, used mainly by the Replicated installer
3. `.wave` app bundle

All three of them are generated by the CLI. The simplest example of how to generate the bundle would
be:

```sh
cd /path/to/my-app
h2o bundle \
  --generate-dockerfile \
  --docker-use-buildkit \
  --docker-base-image 524466471676.dkr.ecr.us-east-1.amazonaws.com/q8s/launcher:v0.23.0-gpu38 \
  --generate-helm-charts \
  --helm-chart-version 0.1.0
```

This command generates:

- Dockerfile in `<APP_NAME>.<APP_VERSION>.Dockerfile`
    - `<APP_NAME>` is set to `app.name` from `app.toml`
    - `<APP_VERSION>` is set to `app.version` from `app.toml`
- `helm/`
    - directory containing the helm charts for the app registration
- app bundle in `<APP_NAME>.<APP_VERSION>.wave`
    - `<APP_NAME>` is set to `app.name` from `app.toml`
    - `<APP_VERSION>` is set to `app.version` from `app.toml`

#### BuildKit

[Docker BuildKit](https://docs.docker.com/build/buildkit/) is enabled when `--docker-use-buildkit`
is specified. It is recommended to have this enabled as it can lead to shorter build times and
allows use of more advanced features. However, in certain cases, such as old docker version,
BuildKit is not available and it should be disabled (by removing the flag).

#### OS packages

If additional OS packages are required, they should be specified in `packages.txt`. This file should
contain newline-separated list of OS packages required by the app. Presence of the file is
discovered automatically, no extra flag is required.

#### Custom scripts

Custom scripts can be supplied to the `h2o bundle` command, that are executed during the docker
image build. These scripts are meant to do initialization, such as download of external files.

```sh
cd /path/to/my-app
h2o bundle \
  --generate-dockerfile \
  --docker-use-buildkit \
  --docker-base-image 524466471676.dkr.ecr.us-east-1.amazonaws.com/q8s/launcher:v0.23.0-gpu38 \
  --generate-helm-charts \
  --helm-chart-version 0.1.0 \
  --docker-pre-install-script path/to/pre-install.sh \
  --docker-post-install-script path/to/post-install.sh \
```

- if `--docker-pre-install-script` is specified, the given script is executed before installing the
  App requirements

- if `--docker-post-install-script` is specified, the given script is executed at the very end of
  the docker image build process


#### Including additional files

In cases when additional files must be copied to the docker image the `--docker-include` flag can be
used. This flag should point to a file or directory that should be copied to the docker image. This
flag can be specified multiple times.

```sh
cd /path/to/my-app
h2o bundle \
  --generate-dockerfile \
  --docker-use-buildkit \
  --docker-base-image 524466471676.dkr.ecr.us-east-1.amazonaws.com/q8s/launcher:v0.23.0-gpu38 \
  --generate-helm-charts \
  --helm-chart-version 0.1.0 \
  --docker-include deps
```

The command above will create a docker image, that'll contain the `deps` dir. Directory is placed on
the same relative path (relative to the app) as it has been on the host. Absolute path is considered
internal detail of the implementation.

#### Specifying custom environment variables

Custom environment variables can be specified using `--docker-env`. The value of the flag should be
in format `MY_VAR=MY_VALUE`. This example will set env var `MY_VAR` to value of `MY_VALUE`. The
variable name cannot contain `=` sign, on the otherhand, the value can contain `=` sign. The flag
can be specified multiple times.

```sh
cd /path/to/my-app
h2o bundle \
  --generate-dockerfile \
  --docker-use-buildkit \
  --docker-base-image 524466471676.dkr.ecr.us-east-1.amazonaws.com/q8s/launcher:v0.23.0-gpu38 \
  --generate-helm-charts \
  --helm-chart-version 0.1.0 \
  --docker-env FOO=BAR \
  --docker-env MY_VAR=MY_VALUE
```

The command above will create a docker image, that contains statements specifying two env vars -
`FOO` and `MY_VAR`.

#### Additional options

All options that can be specified can be discovered using `h2o bundle --help`.

### Publishing an app for others to see and launch

To publish an app to the platform, just run `h2o bundle import` (or `h2o bundle` and `h2o app import <bundle>`)
in your app source directory.
This command will automatically package your current directory into a `.wave` bundle and import it
into the platform.

If you set the visibility to `ALL_USERS` (via the `-v` flag), others will be able use `h2o app run`
or the H2OAIC App Store to launch the app.

**Note:**
The name-version combination from your `app.toml` has to be unique and the platform will reject
the request if such combination already exists. Therefore, you need to update the name or version
in `app.toml` before each consecutive import command.


```sh
$ h2o bundle import -v ALL_USERS
ID              bcd543210-1111-1111-1111-0123456789ab
Title           Peak
Version         0.1.2
Created At      2020-10-13 06:28:03.050226 +0000 UTC
Updated At      2020-10-13 06:28:03.050226 +0000 UTC
Owner           user1@h2o.ai
Visibility      ALL_USERS
Description     Forecast of COVID-19 spread
Tags
```

### Running an app under development

For faster development, running the `h2o bundle test` command in your app source directory will
package your current directory, deploy it and automatically start tailing the logs. On `ctrl-c` the instance will be terminated and the app will be deleted. The CLI will append a `bundle-test` suffix to the version in order to ensure resources are cleaned up.

```sh
$ h2o bundle test
Waiting for instance to start...

ID 	487e6f42-d5e8-4e05-a835-6d73f1488240
URL	https://487e6f42-d5e8-4e05-a835-6d73f1488240.cloud.h2o.ai

Use the URL above to open the running App Instance.
Terminate the instance and delete the app bundle with ctrl-c.

To continue and view the instance logs, press enter:
```

### Deploying multiple versions of an app during development

To immediately run your current app source in the platform, just run `h2o bundle deploy` in your app
source directory.
This command will automatically package your current directory into a `.wave` bundle, import it into
the platform, and run it.

In the output you will be able to find a URL where you can reach the instance, or visit
the "My Instances" in the UI.

**Note:**
The CLI will automatically append a suffix to the version in your `app.toml` so that you can keep
iterating on your app without worrying about version conflicts, just don't forget to clean up old
instances/versions. Unlike `h2o bundle test`, resource intensive features like resource volume persistence (enabled by setting `ResourceVolumeSize`) are disabled.


```sh
$ h2o bundle deploy
ID              bcd543210-1111-1111-1111-0123456789ab
Title           Peak
Version         0.1.2-20201013062803
Created At      2020-10-13 06:28:03.050226 +0000 UTC
Updated At      2020-10-13 06:28:03.050226 +0000 UTC
Owner           user1@h2o.ai
Visibility      PRIVATE
Description     Forecast of COVID-19 spread
Tags
ID  22222222-3333-4444-5555-666666666666
URL https://22222222-3333-4444-5555-666666666666.cloud.h2o.ai
```

### Running the app in a cloud-like environment locally

To execute your app in an environment equivalent to that of the platform but on your local machine,
just run `h2o exec` in your app source directory.
This will package the app in a temporary `.wave` bundle and launch it locally using our platform
docker image.

Note that this requires that you have docker installed and that you have access to the docker image.

Then navigate to `http://localhost:10101/<your main route>`.

```sh
$ h2o exec
{"level":"info","log_level":"debug","url":"file:///wave_bundle/q-peak.0.1.2.wave","app_root":"/app","venv_root":"/resources","server_path":"/wave/wave","py_module":"peak","tmp":"/tmp","startup_server":true,"version":"latest-20200929","time":"2020-10-13T06:42:21Z","message":"configuration"}
{"level":"info","port":":10101","time":"2020-10-13T06:42:21Z","message":"starting launcher server"}
{"level":"info","executable":"/wave/wave","time":"2020-10-13T06:42:21Z","message":"wave executable found"}
...
```

### Updating app visibility

The `h2o app update <appId> -v <visbility>` command can be used to modify an existing app's visibility.

Authors who publish a new version of an app may want to de-list the old version.
The preferred method to de-list previous versions is to modify the visibility setting to `PRIVATE`.
In fact, it is not possible to fully delete an app if there are instances running and doing so might
affect them.

### Downloading an app

The `h2o app download <appID> [<path>] --extract` command can be used to download an app. Authors who publish a version of an app may want to download the app bundle. An optional `--extract` flag will extract the bundle after download. By default the path will be the `name.version` of the app, if using the extract flag it will extract to a folder with the same name.

The app id can be copied from the **App Details**, **My Apps**, or **Admin Apps** page.

### Setting a default launch profile for the app

First run `h2o env launch-profiles list` to list the available launch profiles. You can then run `h2o app set-default-profile <profile-name>` to set the default launch profile for a particular app. Alternatively, you can also set the default launch profile while importing the app by running `h2o app import --profile <profile-name>`. 


## App instances

The `h2o instance [get, list, logs, status, suspend, resume, terminate, update]` commands let you see and,
when authorized, manage available app instances.

### Getting instance logs

To see logs of a running instance, just run `h2o instance logs <instanceId>`; use the flag
`-f` (`--follow`) to tail the log.

```sh
- $ h2o instance logs 22222222-3333-4444-5555-666666666666
- ...
- 2020/10/27 16:16:34 #
- 2020/10/27 16:16:34 # ┌─────────────────────────┐
- 2020/10/27 16:16:34 # │  ┌    ┌ ┌──┐ ┌  ┌ ┌──┐  │ H2O Wave
- 2020/10/27 16:16:34 # │  │ ┌──┘ │──│ │  │ └┐    │ (version) (build)
- 2020/10/27 16:16:34 # │  └─┘    ┘  ┘ └──┘  └─┘  │ © 2020 H2O.ai, Inc.
- 2020/10/27 16:16:34 # └─────────────────────────┘
- 2020/10/27 16:16:34 #
- 2020/10/27 16:16:34 # {"address":":10101","t":"listen","webroot":"/wave/www"}
- 2020/10/27 16:16:34 # {"host":"ws://127.0.0.1:10102","route":"/","t":"relay"}
- ...


### Getting instance status

To see some details about the (K8s) runtime status of a running instance, just run
`h2o instance status <instanceId>`.

```sh
$ h2o instance status 22222222-3333-4444-5555-666666666666
Status       	CrashLoopBackOff
Reason       	Error
Message
Exit Code    	1
Restart Count	1
Last Failure 	<UNKNOWN>
```

### Updating instance visibility

The `h2o instance update <instanceId> -v <visbility>` command, much like the `app` version, can be
used to modify an existing running instance's visibility setting.

## App tags

The `h2o tag [assign, get, list, remove, update]` commands let you see and, when authorized, manage
available app tags.

The alias commands let you (if you are an admin) see and manage available aliases.
(see [Basic concepts](#app-alias) for details on Aliases and their
attributes).

## Secrets

The `h2o secret [create, get, delete, update, list]` commands let you see and, when authorized,
manage available secrets.

Many commands allow specifying the scope of the secret(s) via the `-v` and `-p` options for
the `visibility` and `parent` attributes, respectively. The value of `parent` differs based
on `visibility`:

- `PRIVATE`: empty
- `ALL_USERS`: empty;
- `APP`: URN referring to the corresponding app name, in the format `app:<app name>`.

**Note:**

Only admins can currently interact with secrets with visibility `APP`,
see [Authorization](#secret-authorization-for-users-with-full-access).



See [Basic concepts](#app-secret) for details on Secrets and their
attributes.

### Creating and updating private secrets

`h2o secret [create, update] <secretName> [--from-literal=key=value] [--from-file=key=/path/file]`

Where,

- `<secretName>` is the name of the secret
- `--from-literal=key=value` specifies that the value of `key` in the secret should be `value`
- `--from-file=key=/path/file` specifies that the value of `key` in the secret should be the
  contents of the file at path `/path/file`

Creates or updates a `PRIVATE` secret. Based
on [Kubernetes Secrets](https://kubernetes.io/docs/tasks/configmap-secret/managing-secret-using-kubectl/)

Secret names have a maximum length of 63 characters and are validated using the following regex
`^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$`.

Secret keys have a maximum length of 253 characters and must consist
of `alphanumeric characters, -, _ or .`.

```sh
$ h2o secret create secret-name --from-literal=key=value --from-file=myfile=secret_file.txt
Name      	secret-name
Visibility	PRIVATE
Keys      	key, myfile
```

Updating a secret will replace the current keys and data stored in the secret.

```sh
$ h2o secret update secret-name --from-literal=key=newValue --from-file=myfile=secret_file.txt
Name      	secret-name
Visibility	PRIVATE
Keys      	key, myfile
```

### Creating and updating public and app-scoped secrets

`h2o admin secret [create, update] [-v <visibility>] [-p <parent>] <secretName>`

Creates or updates (if you are an admin) an `ALL_USERS` or `APP` secret, similar to
the [section above](#creating-and-updating-private-secrets).

There can be multiple secrets with the same name with different scopes. To avoid ambiguity and choose the specific secret you want to update, `h2o admin secret update` accepts a `visibility` flag.

See [Secrets](#secrets) for more details on the `visibility` flag for secrets.

### Deleting private secrets

`h2o secret delete <secretName>`

Where,

- `<secretName>` is the name of the secret

Deletes a `PRIVATE` secret. Secrets cannot be deleted if they are currently in use by any apps.

### Deleting public and app secrets

`h2o admin secret delete [-v <visibility>] [-p <parent>] <secretName>`

Deletes (if you are an admin) an `ALL_USERS` or `APP` secret similar to
the [section above](#deleting-private-secrets). Secrets cannot be deleted if they are currently in
use by any apps.

### Listing secrets

`h2o secret list [-v <visibility>]` <br/> `h2o admin secret list [-v <visibility>] [-p <parent>]`

Lists existing secrets.

The default visibility for non-admin users is `UNSPECIFIED`, which will list all secrets the user
has access to;
see [Authorization](#secret-authorization-for-users-with-full-access).
secrets attributes.

The default visibility for admins is `ALL_USERS`.

## Tags

The tag commands let you see and manage available tags.
See [Basic concepts](#app-tag) for details on Tags and their attributes.

### Listing tags

`h2o tag list`

Lists relevant details about the current tags.

### Getting tag information

`h2o tag get <tag id>`

Where,

- `<tag id>` is the tag ID (from `admin tag list`)

Displays additional information about a specific tag.

### Assigning tags

`h2o tag assign <tag id> <app id>`

Where,

- `<tag id>` is the tag ID (from `tag list` or `admin tag list`)
- `<app id>` is the application ID (from `app list`, `app list -a`, or `admin app list`)

Assigns a specific tag to a specific version of an application.

### Removing tags

`h2o tag remove <tag id> <app id>`

Where,

- `<tag id>` is the tag ID (from `tag list` or `admin tag list`)
- `<app id>` is the application ID (from `app list`, `app list -a`, or `admin app list`)

Unassigns a specific tag from a specific version of an application.

### Updating tags

`h2o tag update <tag id> [-n <name> | --name <name>] [-c <color> | --color <color>] [-t <title> | --title <title>] [-d <description> | --description <description>] [-v <visitor role> | --visitor-roles <visitor role>] [-a <visitor admin role> | --admin-roles <visitor admin role>]`

Where,

- `<name>` is the canonical name for the tag, to be used in [app.toml](#apptoml)
- `<visitor role>` is the OIDC role a user must have to see apps assigned this tag
- `<visitor admin role>` is the OIDC role of users who may assign this tag to apps
- `<title>>` is the text that appears on the app card in the web interface
- `<color>` is the hex value used to colorize the tag in the web interface

Mutates an existing tag.

### Creating a category tag

`h2o admin tag create --name <category tag name> --title <category tag title>`

Where,

- `<category tag name>` is the canonical name for the tag, to be used in Helm values (`values.yaml` file)
- `<category tag title>` is the text that appears on the **Categories** section of the App Store web interface

After creating a new category tag using H2O CLI, add the name of the category tag to the `values.yaml` file.

  ```
categoryTags: ["AI_FOR_GOOD", "EDA", "MACHINE_LEARNING", "EXPLAINABILITY", "COMP_VISION", "FORECASTING", "NLP", "UNSUPERVISED_LEARNING", "FEDGOV", "FINSERV", "HEALTHCARE", "MANUFACTURING", "MARKETING", "RETAIL", "TELECOM", "APP_DEV"]
  ```

**Note:**
Tags, including category tags, may only be created by an administrator.


## Aliases

`h2o admin alias [assign, create, delete, get, list, promote]`

The alias commands let you (if you are an admin) see and manage available aliases.
See [Basic concepts](#app-alias) for details on Aliases and their attributes.

### Listing aliases

`h2o admin alias list`

Lists relevant details about the current aliases.

### Creating aliases

`h2o admin alias create <alias> [<instance id>] [<primary>]`

Where,

- `<alias>` is the name of the alias (this is what determines the URL, e.g., alias `hello` results in URL `hello.cloud.h2o.ai`)
- `<instance id>` (optional) is the instance ID of the instance to assign this tag to at create time
  (from `instance list`, `instance list -a`, or `admin instance list`)

- `<primary>` (optional) is `true` or `false` depending on whether you want the tag to be marked primary at creation time or not

Creates an alias and (optionally) assigns it to an instance.

```shell
$ h2o admin alias create hello
ID         	11111111-2222-3333-4444-555555555555
Alias      	hello
Instance ID	00000000-0000-0000-0000-000000000000
Primary    	false
Created At 	2022-03-02 12:52:08.900656 +0000 UTC
Updated At 	2022-03-02 12:52:08.900656 +0000 UTC
```

### Getting alias information

`h2o admin alias get <alias id|name>`

Where,

- `<alias id|name>` is the alias name or ID (from `admin alias list`)

Displays additional information about a specific alias.

```shell
$ h2o admin alias get hello
ID         	11111111-2222-3333-4444-555555555555
Alias      	hello
Instance ID	00000000-0000-0000-0000-000000000000
Primary    	false
...
```

### Assigning aliases

`h2o admin alias assign <alias id|name> [<instance id>] [<primary>]`

Where,

- `<alias id|name>` is the alias name or ID (from `admin alias list`)
- `<instance id>` (optional) is the instance ID of the instance to assign this tag to (from `instance list`, `instance list -a`, or `admin instance list`); if empty, the alias will be unassigned
- `<primary>` (optional) is `true` or `false` depending on whether you want the tag to be marked `primary` at assign time or not; this parameter will be ignored if `<instance id>` is empty.

Assigns an alias to an instance and optionally makes it `primary`. By default, assigning an alias cleans the `primary` bit.

```shell
$ h2o admin alias assign hello 22222222-3333-4444-5555-666666666666
ID         	11111111-2222-3333-4444-555555555555
Alias      	hello
Instance ID	22222222-3333-4444-5555-666666666666
Primary    	false
...
$ h2o admin alias assign hello 22222222-3333-4444-5555-666666666666 true
ID         	11111111-2222-3333-4444-555555555555
Alias      	hello
Instance ID	22222222-3333-4444-5555-666666666666
Primary    	true
...
$ h2o admin alias assign hello
ID         	11111111-2222-3333-4444-555555555555
Alias      	hello
Instance ID
Primary    	false
...
```

Note that there can be a slight delay before the change gets propagated.

### Promoting aliases to primary

`h2o admin alias promote <alias id|name>`

Where,

- `<alias id|name>` is the alias name or ID (from `admin alias list`)

Promotes an alias to primary for the corresponding instance.

```shell
$ h2o admin alias promote hello
ID         	11111111-2222-3333-4444-555555555555
Alias      	hello
Instance ID	22222222-3333-4444-5555-666666666666
Primary    	true
...
```

Note that there can be a slight delay before the change gets propagated.

As per [Basic concepts](#app-alias),
accessing the instance via other aliases or via its UUID URL will result in a HTTP `302` redirect to the primary alias.
If there was another alias marked primary for the same instance, its `primary` bit will be unset.

### Deleting aliases

`h2o admin alias delete <alias id|name>`

Where,

- `<alias id|name>` is the alias name or ID (from `admin alias list`)

Deletes an alias.

Note that there can be a slight delay before the change gets propagated.

\newpage


# Importing apps

This tutorial guides you through importing a Wave app or other Python apps to the App Store, so you can iteratively develop and prototype the app and then publish it to users.

## Before you begin

Before you begin, you must have the following artifacts:

- Access to the H2O AI Cloud App Store
- App upload permissions
- The Wave or Python app source code of the app that you wish to upload

## App configurations

First, ensure that your app source code is bundled and ready for import. For more information about app bundle structure, see [Developer guide](#developer-guide).

There are a few more files that are required before you go ahead and import to the App Store.

1. **`requirements.txt` file **- Create a file called `requirements.txt` to indicate which packages and versions to install in the Python virtual environment when running the app on the App Store.

   ```py title="sample requirements.txt file"
   altair==4.1.0
   h2o-wave==0.20.0
   ```

**Note:**
   For production use cases, always pin your libraries and their transitive dependencies in the `requirements.text` with a specific version to ensure that your app does not break or get unexpected behavior when a new version of a library is available on PyPi. For more information, see [Requirements File Format](https://pip.pypa.io/en/stable/reference/requirements-file-format/) in the Pip documentation.
   

2. **`.appignore` file** - Create a file named `.appignore` to reduce the size of your Wave app bundle. This file allows you to specify which files or directories to exclude, ensuring that only essential components are included.

3. **`app.toml` file** - Create a new file called `app.toml`. This is a configuration file that tells the App Store how to present the app to end users and how to run it. It contains details such as the location path to the entry point of the source code, app title, app secrets, and indicates any special configuration for running the app such as whether or not users need to log in to use the app, should there be GPUs, etc. 

   Setting up and configuring other Python applications follows a similar process, with only a few adjustments required in the `app.toml` file. Here's an example with the key differences to illustrate:






**Wave apps**

The `app.toml` file for Wave apps :
```python
toml title="sample app.toml file"
[App]
Name = "ai.h2o.wave.my-first-app"
Version = "0.0.1"
Title = "Hello, World!"
Description = "Show a card to the user explaining this is my first app."

[Runtime]
Module = "app"
```


**Python apps**

The `app.toml` file for Python apps :
```python
toml title="sample app.toml file for Python apps"
[App]
Name = "ai.h2o.demo.streamlit"
Version = "0.0.1"
Title = "Streamlit Demo App"
Description = "Publish and use Streamlit in H2O's AI App Store"
LongDescription = "README.md"
InstanceLifecycle = "Managed"
Tags = ["DEMO", "APP_DEV"]

[Runtime]
Module = "src.entrypoint"
AppMode = "python" # Required for AI App Store to know this is not a Wave app
Port = 8501 # Required to successfully access this app
RoutingMode = "DOMAIN" # "BASE_URL"
```



     

    
**Notes:**

- To see the full list of configuration options for this file, see [Developer guide - app.toml](developer-guide#apptoml).

- `Module` is the name name of the python module that is the entrypoint of the app (relative to the `app.toml` file). If you have all your source code in a folder called `src` and your main app file is called `app.py`, the value for `Module` would be `src.app`. For more information about Python Modules, see the [Python documentation](https://docs.python.org/3/using/cmdline.html#cmdoption-m).


4. **`app.py` file** - This is the main source code file for your Python app.

   ```python title="sample app.py file"
   from h2o_wave import Q, app, main, ui  # noqa: F401

   @app("/")
   async def serve(q: Q):
       q.page["lorem"] = ui.markdown_card(
           box="1 1 2 2",
           title="Lorem Ipsum",
           content="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin id blandit nunc.",
       )
       await q.page.save()
   ```

### AI App Store metadata

#### LongDescription  

`LongDescription` is a path to a file, relative to the archive root, containing an additional multi-line [Markdown-formatted](https://daringfireball.net/projects/markdown/) description of the app. The file typically includes a brief description of the app, its features, target audience, and information on sending feedback. Limiting the content of this file to bulleted lists (`*`), H3 headings (`###`), and hyperlinks (`[]()`) is recommended. The `LongDescription` goes to the **App details** section of a particular app.

#### Images

The static folder of the app bundle contains screenshots (files starting with `screenshot`) of the app's features, graphs, tables, or the application flow. These screenshots are displayed in the **App details** section.
#### Icon

The static folder of the app bundle contains the app icon (a PNG or JPEG file starting with `icon` or the application name), which appears on the particular app tile. 
The following is a list of sizing guidelines for app icons:

- The icon must be square with the same height and width.
- The icon must be at least 400 pixels wide and no larger than 1000 pixels.
- The file size of the icon must be less than 100KB.
## Importing and publishing an app

You can import your Wave or Python app to the App Store using one of the following methods. The easiest and fastest way to deploy, run, and manage your app is using the H2O CLI. However, you can also use the GUI if you prefer.

- [Using H2O CLI (recommended)](#using-h2o-cli)
- [Using GUI (alternative)](#using-the-gui)

**Note:**
To try importing and publishing an app, you can download the [sample bundled app](upload-wave-app-guide/ai.h2o.wave.hello-world.0.0.1.wave "download") and follow the below instructions to deploy it on your H2O cloud.


### Using H2O CLI

1. Set up and configure [H2O CLI](#cli).

2. Navigate to your app's source directory on the command line.

3. Run the following command to immediately import and run your app on the App Store.

   ```
   h2o bundle deploy --generate-version=false
   ```

**Note:**
   For more information, see [Running an app under development](cli#running-an-app-under-development) or [Publishing an app for others to see and launch](cli#publishing-an-app-for-others-to-see-and-launch).
   

   > Once the app is imported, you will see it listed on the **MY APPS** page of the App Store. This page will list all the applications that you own. Now that the app has been imported successfully, you can update the visibility of the app, view/update app details, run it, or delete the app [using the H2O CLI](cli#apps) or directly via the **MY APPs** page.

**Note:**
   You can use `.appignore` to ignore the unnecessary files while bundling your app and ensure that you don't have any unwanted files being bundled. For more information, see [`app.toml`](#apptoml) in App Bundle Structure.
   

4. Access the URL provided on the CLI on a browser window to see your app.

   ```sh
   $ h2o bundle deploy --generate-version=false
   ...
   URL https://22222222-3333-4444-5555-666666666666.cloud.h2o.ai
   ```

5. To publish it and make it available to all users, run the following command on the CLI.

   ```sh
   $ h2o app update -v
   ```

**Note:**
   For more information, see [Updating app visibility](cli#updating-app-visibility)

6. Access the App Store on a browser window.

   You will see that your app is now published on the App Store page and is ready to be used by other users on the platform!

   ![](upload-wave-app-guide/h2o_appstore_screen.png)

### Using the GUI

**Optionally**, you can also run this app using the H2O AI Cloud user interface.

1. Click **MY APPS** on the H2O AI Cloud home page.

   ![](upload-wave-app-guide/my_apps.png)

2. Click **Import App**. You will be prompted to upload your application as a .zip file.

   ![](upload-wave-app-guide/import_new_app.png)

3. You can open up the folder that contains your application source code and compress the folder into a `.zip` file. If you are using MacOS, select the relevant files and folders that you wish to compress individually instead of selecting and compressing the entire /root folder. For example: 

   ![](upload-wave-app-guide/compress_file.png)

**Note:**
    **Alternatively**, you can compress your app files and folders and create an app bundle. Navigate to your app's root directory and run `h2o bundle` (you do not need to have configured the H2O CLI to run this command).
   

4. Upload the .zip file that you just created and select the required **Visibility** for the app. You can select `PRIVATE` when first uploading the app to do some light testing before publishing it to other users.

   - `PRIVATE`: This setting makes the app visible to only you (the app owner). The app will not be visible or searchable on the App Store.

   - `ALL_USERS`: This setting makes the app visible on the app store to all users.

   > Once the app is imported, you will see it listed on the **MY APPS** page. This page will list all the applications that you own. Now that the app has been imported successfully, you can update the visibility of the app from here, view/update app details, run it, or delete the app.

5. At this point, the app is still `PRIVATE`.

   To publish it to all users, on the **MY APPS** page, click **Visibility**.

   ![](upload-wave-app-guide/update_app_visibility.png)

6. Select `ALL_USERS` and click **Update**.

7. Next, click on **APP STORE** on the top navigation pane.

   ![](upload-wave-app-guide/h2o_appstore_screen.png)

   You will see that your app is now published on the App Store page and is ready to be used!

## App Deprecation

We consider an app or a version of an app as 'deprecated' when the developers discourage its use because it leads to errors or new versions of the particular app exist.

The owner of the app can't delete an app version if there are running instances. Our process is to make the old app version private, so the users will not be able to launch new instances. Then eventually, the old app version will get deleted as there are no more instances.

## Summary

This tutorial walked you through:

- Setting up your Wave or Python application bundle
- Importing and publishing an app using the H2O CLI (recommended option)
- Importing and publishing an app using the App Store GUI (alternative option)

**Note:** More resources
You can also check out the following blog post and documentation for more information.

- **(Blog)** [Setting Up Your Local Machine for H2O AI Cloud Wave App Development](https://h2o.ai/blog/setting-up-your-local-machine-for-h2o-ai-cloud-wave-app-development/)
- **(Documentation)** [Wave Get Started Tutorial](https://wave.h2o.ai/docs/getting-started)


\newpage

# Apps with a managed instance lifecycle

By default, users of the App Store are encouraged via the UI to create and run their own instance of an app, but can also share instances and see instances from other users with a few button clicks. 

However, as an app owner, you have the ability to add a restriction that only allows the app owner (you) or adminstrators to create a new instance of this specific app. This is called an 'app with managed instance lifecycle' (while the default behavior is called on 'demand instance lifecycle').  

Users are directed to visit a particular (your) instance of the app in the UI instead of being given the option to create their own. This is ideal for apps that either have no state or are explicitly built to be multi-user.

This feature allows for an easier user experience, as users don't have to always wait to start their own instance. Additionally, it saves costs by sharing resources when appropriate.


1. All apps that are published on the H2O AI Cloud App Store require an `app.toml` file, which tells the App Store how the app works and how to display it to end users.

2. The `[App]` section of the [app.toml file](#apptoml) has a parameter called `InstanceLifecycle`. This is an optional parameter and by default the value will be `ON_DEMAND`, which means every user creates/runs their own instance. Change this value to `MANAGED`.

3. Next, use the UI or CLI to upload the app.
  ```
  h2o bundle import -v ALL_USERS
  ```

  When the owner visits the app in the App Store, they will have the option to run the app. Other users who have permission to see the app will be able to see the app details, but will not be able to create a new instance of the app. Below, we see that Michelle is the owner of the app and Doug is visiting the app details page.

  **App owner view**

  ![](managed-instance-guide/managed-instance-app-owner.png)

  **App visitor view**

  ![](managed-instance-guide/managed-instance-app-visitor.png)

4. As the app owner, you can run the app using the UI or you can use the following CLI command.
  ```
  h2o app run <APP_ID> -v ALL_USERS
  ```

  Once at least one version of the app exists, all `FULL_ACCESS` users and [visitors](#authorization-for-visitors) that are able to see the app will have a **Visit** button that will open up the newest app instance.

  ![](managed-instance-guide/managed-instance-running-app.png).

\newpage

# Overview

H2O AI Hybrid Cloud (H2OAIC) App Store is a scalable, light-weight system for managing and scheduling of Wave apps in Kubernetes.

![](./assets/arch.svg)

The H2OAIC **App Store Server** is the heart of the H2O AI Hybrid Cloud. It is responsible for managing applications
and their instances. It consists of the following major components:

* **App Store Frontend** - Provides a web interface for interacting with the App Store
* **CLI** - Provides a command line interface for interacting with the App Store  
* **App Repository** - Manages app bundles and metadata, relying on Blob storage and PostgreSQL
* **Scheduler** - Interfaces with Kubernetes to start and manage running applications and secrets 
* **API** - Provides access to App Repository and Scheduler for web UI and CLI
* **Router** - Handles authorization and routing of incoming traffic to app instances

  
The H2OAIC **Launcher** manages the runtime environment and lifecycle of a single app instance.

## App Store Server

The App Store runs as a single deployment within Kubernetes and provides the App Store frontend, App Repository, Scheduler, API Server, and Router services.
It can run with multiple replicas in a highly-available manner to ensure zero downtime updates and 
fault tolerance as well as performance at scale.

### App Store Frontend

The App Store frontend is primary user interface for users of the H2O AI Hybrid Cloud, providing users an easy-to-use interface 
for managing apps and app instances.
With built in support for visitor access, administrators can tailor which apps specific users can see/launch.

### CLI

The H2OAIC provides a CLI that users can utilize to manage their workloads and serves as the primary tool that app
developers will use to publish apps into the App Store. Users can list apps, launch and manage instances, as well as
get access to instance diagnostics like logs and instance status. See the [CLI documentation](#cli)

### App Repository

Management of app bundles (app source packages uploaded by app developers) and metadata is handled by the
App Repository. 
It is a straightforward web application which uses PostgreSQL to store and query app metadata extracted
from `app.toml` (see [Developer guide](#developer-guide)), 
including app tags. It utilizes a Blob/Object store to persist the bundles and other
files, such as icons and screenshots.

It provides two notable HTTP handlers: one for importing a new bundle and one for downloading 
a bundle for execution by the [Launcher](#launcher).

### Router

Requests to app instances pass through the App Store server, enforcing their authorization settings; 
see [Instance access controls](#wave-application-instance-access-controls)

The App Store uses the hostname of the incoming request to properly route requests (e.g., `<instance>.wave.h2o.ai`),
so it's important to provide a setup where a wildcard DNS record and TLS certificate may be used;
see [Deployment](#appstore-server-deployment) for details.

When the App Store receives a request for an app instance, it first consults Scheduler to locate 
the instance within Kubernetes, determines and enforces access restrictions, before proxying 
the request to the destination Kubernetes service. 
All requests, even websocket requests, are authenticated before they are passed to the running app instance.
The visibility level of an instance may be updated at any time using the CLI.


### Scheduler

The App Store utilizes Helm to launch and manage running Wave app instances without requiring 
an external database to maintain app instance state, i.e., the Kubernetes API is the only place
storing the app instance state.
Administrators can manage a list of eligible docker images that Wave apps can be launched in using
the `RuntimeVersions` configuration.
The scheduler can optionally mount Kubernetes secrets, attach Persistent Volumes or ensure GPU allocation 
for apps that require it. 
It can also read instance status and logs.

The scheduler can be configured to limit the number of instances per user, the number of published app versions per user, and more.

The configuration options may be either set within the `[Scheduler]` section of the server
configuration `ConfigMap` TOML, or set via environment variables.

The environment variable names in the table below need to be prefixed with `H2O_CLOUD_SCHEDULER_`:

| TOML Key |<br/> Environment Variable | Description | Default |
| --- | --- | --- |
| KubeConfig<br/>`KUBE_CONFIG` | Optional; specifies a path to a Kubernetes configuration file for cluster access; empty means in-cluster access | (empty)
| Namespace<br/>`NAMESPACE` | Kubernetes namespace to schedule apps within | `default`
| HelmAssetDir<br/>`HELM_ASSET_DIR` | Directory to extract Helm assets to for scheduling apps | `var/lib/h2oq8s/helm`
| ServiceType<br/>`SERVICE_TYPE` | Kubernetes service type to create when scheduling apps | `ClusterIP`
| StorageClass<br/>`STORAGE_CLASS` | Kubernetes PersistentVolume StorageClass to attach to apps requiring persistent storage | (empty)
| WriteTimeoutSeconds<br/>`WRITE_TIMEOUT_SECONDS` | Default timeout for running, terminating and updating instances | `300`
| ResourcePath<br/>`RESOURCE_PATH` | Path within app containers to mount a PersistentVolume at if `ResourceVolumeSize` is specified in the app.toml | `/resources`
| InstanceLimit<br/>`INSTANCE_LIMIT` | Maximum number of instances a full access user may have running | `10`
| VisitorInstanceLimit<br/>`VISITOR_INSTANCE_LIMIT` | Maximum number of instances users without full access may have running | `5`
| AppVersionLimit<br/>`APP_VERSION_LIMIT` | Maximum number of apps a full-access user may import | `10`
| AppServiceAccount<br/>`APP_SERVICE_ACCOUNT` | Kubernetes ServiceAccount to be used by apps | (empty)
| AllowedGPUTypes<br/>`ALLOWED_GPU_TYPES` | Names of allowed GPU types; empty means GPU support is disabled; should match existing values of the `hac.h2o.ai/accelerator` K8s node label | (empty)

The default app resources may be either set within the `[Scheduler.DefaultRuntimeLimit]` section of the server
configuration `ConfigMap` TOML, or set via environment variables.

The environment variable names in the table below need to be prefixed with `H2O_CLOUD_SCHEDULER_DEFAULT_RUNTIME_LIMIT_`:

 | TOML Key /<br/>Environment Variable | Description | Default |
 | --- | --- | --- |
 | MemoryLimit<br/>`MEMORY_LIMIT` | Default memory reservation for apps if unspecified by the app; needs to conform to the  [K8s resource model](#resource-quantities) | `2Gi`
 | MemoryReservation<br/>`MEMORY_RESERVATION` | Default memory limit for apps if unspecified by the app; needs to conform to the  [K8s resource model](#resource-quantities) | `512Mi`
 | CPULimit<br/>`CPU_LIMIT` | Default CPU reservation for apps if unspecified by the app;needs to conform to the [K8s resource model](#resource-quantities) | (empty)
 | CPUReservation<br/>`CPU_RESERVATION` | Default CPU limit for apps if unspecified by the app; needs to conform to the [K8s resource model](#resource-quantities) | (empty)
 | GPUType<br/>`GPU_TYPE` | GPU type that should be assigned by default; empty means any, delegating to the K8s scheduler; should match existing values of the `hac.h2o.ai/accelerator` K8s node label | (empty)

### Launcher

The H2OAIC Launcher is the runtime environment responsible for installing dependencies, starting the waved server, and managing the lifecycle of the app.

When an app instance is launched, the launcher will download the `.wave` bundle from the App Store server. This will show the user of the instance a `Sourcing The Wave Application` message in the browser.

Next it will install any python, and system packages that the app requires. This will show the user of the instance a `Installing Application Dependencies` message in the browser.

After the necessary packages have been installed in the python virtual environment, the `waved` server will start, and the lifecycle of the wave python code will begin. 
Both the stdout of the waved server and wave application with be piped to stdout. If either of the `waved` or the application code fail to start the HAC launcher will exit.

In the server configuration you can specify multiple `RuntimeVersion`s, effectively docker images,  that can be used to launch apps.

```toml
[[RuntimeVersion]]
Name = 'deb10_py37_wlatest'
Image = "container-registry/wave-launcher:latest"
Default = true
Deprecated = false
```

**Attributes**:

* `RuntimeVersion` (struct) - A base image to be used to launch apps repeated;
  * `Name` (string) - name .
  * `Image` (string) - container image name.
  * `Default` (bool) - pick this image as the default runtime to use
  * `Deprecated` (bool) - prevent new apps from using this image

See [Developer guide](#runtime-environment) for more information on using runtime versions.

## Storage
 
The App Store uses multiple different data stores for various purposes:
1. PostgreSQL - Stores metadata about apps and tags
1. Kubernetes/Helm - Stores state and configuration for running instances
1. Blob Storage - Published Wave app bundles and static app resources like icons & screenshots

## Authentication & Authorization

H2OAI leverages OpenID Connect (OIDC) to authenticate and authorize users within the H2O AI Hybrid Cloud.  This means it's easy to
federate logins with a host of different IdP providers using Keycloak.  See [Authorization](#authorization) for more.

\newpage

# Integrations

The AI App Store provides tight integration with the rest of the H2O AI Hybrid Cloud,
especially [ML Engine management](https://www.h2o.ai/enterprise-support/#enterprise-security) and
[Model management](https://www.h2o.ai/products/h2o-mlops/).

The key principles of these integrations are:

* **Shared user identity (via [OIDC](https://openid.net/connect/))** - all APIs/UIs within 
  the H2O AI Hybrid Cloud support OIDC-based authentication and authorization. 
  This allows users to use a single identity across all the pieces of the H2O AI Hybrid Cloud.
  More importantly, [when configured](https://wave.h2o.ai/docs/security/#single-sign-on), 
  this also allows Wave apps running within the App Store platform to use the users' identity do make
  API calls to the individual components on behalf of the end user.
  Technically this is achieved by leveraging OIDC support in each of the components and configuring
  their respective OIDC clients in such a way that their OIDC access tokens are accepted by the other
  parties as necessary.

* **Shared storage API** - apps running within the App Store platform can use that same storage API
  as the other components of the H2O AI Hybrid Cloud, including access authorization.
  Combined with the shared user identity mentioned above this means that a user can import data via 
  a Wave app (using the API and her identity), utilize it transparently from a Driverless AI engine 
  (which again uses the same API and identity to read the data),
  and process/display the results in another app; all without having to configure connectors, storage 
  resources, or shared workspaces.
  
* **Dependency injection** - apps running within the App Store platform have the references to the
  other H2O AI Hybrid Cloud components injected via their [environment](#apptoml)
  from a [secret](#app-secret).
  This allows for loose coupling of apps and  H2O AI Hybrid Cloud components.
  
Note that these principles can be easily applied to other dependencies, esp. in existing environments,
as long as they support OIDC (access token) authentication.

<div style={{textAlign: 'center'}}>

![](./assets/integrations.svg)

</div>

## Model management

The App Store relies on [H2O MLOps](https://www.h2o.ai/products/h2o-mlops/)
for management of model deployments.

To configure this integration, it is necessary to:

1. Configure MLOps with an OIDC client in the same user pool/realm as the App Store. The client must be 
   able to obtain the `ai.h2o.storage` and `ai.h2o.deploy` scopes for its tokens.

1. Configure the Wave app OIDC client in such a way that it by default obtains the `ai.h2o.storage`
   and `ai.h2o.deploy` scopes for its tokens

1. Configure a shared (`ALL_USERS`) secret with the MLOPs API URL

After this, OIDC-enabled Wave apps can make API calls to MLOPs on behalf of the end user via 
the MLOps Python library.

## ML Engine management
The App Store relies on [H2O AI Engines](https://h2oai.github.io/ai-engine-manager/docs/introduction/)
for management of ML engines, e.g, Driverless AI.

To configure this integration, it is necessary to:

- Configure [AI Engine Manager (AIEM)](https://h2oai.github.io/ai-engine-manager/docs/introduction), with the OIDC platform public client. 
   The client must be able to obtain the `ai.h2o.storage` scope for its tokens.
   Also, make sure that App Store user roles are mapped to a token claim.

   **Example:**
   ```
   auth:
    oidcIssuer: "https://auth.cloud-dev.h2o.ai/auth/realms/hac-dev"
    oidcClientID: "hac-platform-public"
    oidcRoleClaim: "realm_access.roles"
    oidcAdminRole: "admin"
    oidcUserDisplayNameClaim: "preferred_username"
    oidcClientScopes: "openid, offline_access, profile, email"
   ```


After this, OIDC-enabled Wave apps can make API calls to AI Engines/DAI on behalf of the end 
user via the AI Engines/DAI Python client library.

\newpage

# Deployment

## Deploying the App Store

Depending on your DNS/Kubernetes Ingress setup, Deploying AI App Store can be as simple as:

```shell
$ helm upgrade --install h2oaic h2oaicloud \
  --set config.address=https://appstore.<yourdomain> \
  --set config.keycloak_address=http://auth.<yourdomain>
```

This command uses [Helm](http://helm.sh) to deploy the App Store in a hello-world style deployment, 
including all the required dependencies, such as a PostgreSQL database and [Keycloak](https://www.keycloak.org)
as the OIDC provider, ready to go.

Deploying the App Store onto [Minikube](https://minikube.sigs.k8s.io/docs/) or [K3s](https://k3s.io)
is equally simple.

If you have questions or want to discuss deployments in your environment, please contact [sales@h2o.ai](mailto:sales@h2o.ai).

## App Store server deployment

The App Store is designed to run as a highly-available replicated Kubernetes deployment. 
A typical deployment consist of the following resources:

- **Service accounts**

    Two service accounts are required for this deployment: One for the App Store itself, and one for
    Wave apps launched by the App Store. 

- **Kubernetes secrets**

    H2O AI Cloud stores sensitive information such as passwords, client secrets, etc. 
    as Kubernetes secrets. 

- **Kubernetes service**

    Exposes the App Store frontend. This service needs to be exposed via an Ingress to the end users.

- **DNS/TLS**

    The App Store service needs to be exposed under a wildcard DNS entry/TLS cert (e.g., `*.wave.h2o.ai`).
    This is because the App Store uses the subdomains for exposing the individual app instances 
    (i.e., `<instance>.wave.h2o.ai`).

- **ConfigMap**

    The main configuration file for App Store server is defined as
    a [TOML](https://toml.io/en/) file in a Kubernetes ConfigMap.

- **Kubernetes deployment**

    The App Store server can de deployed as a replicated Kubernetes deployment with 
    a single server container in each pod.

- **PostgreSQL**

    The App Store server also requires a PostgreSQL database (>=11.0). Even though it is possible to
    deploy the database directly in Kubernetes (e.g., using the PostgreSQL helm chart),
    it is recommended to use a hosted cloud service.
  
- **Persistent Volume or Object Storage Bucket**

    The App Store needs a storage for large objects, such as app bundles.
    For ease of deployment in test environments, the App Store can use a Kubernetes Persistent Volume. 
    The recommended storage is, however, a cloud object storage bucket, such as AWS S3 or Azure Blob
    Storage.

    
## Wave App deployment

Each Wave application instance is deployed by the App Store using a helm chart.
This helm chart is populated automatically given the values of the app.toml configuration file,
as described in the [Developer guide](#apptoml)

Each app is deployed as a 1-pod **Kubernetes deployment** with **ClusterIP service** and 
optional **Config Map** or **Persistent Volume Claim(s)**.

The pod runs a single generic container image with App Store Launcher as the main process.
See [Overview](#hac-launcher) for details on the Launcher.

\newpage

# Authorization

## Identity provider

H2O AI Hybrid Cloud (H2OAIC) utilizes a compatible OpenID Connect (OIDC) provider, such as Keycloak,
to authenticate and authorize users in both the App Store and Wave apps.
This allows for easy integration and federation with other services like SAML and LDAP.

## User roles

The actions a user may perform within the H2O AI Hybrid Cloud depends on the user's role, as
documented in the [Authorization](#authorization) section of the user guide.

The user's role is based on the OIDC access token claims returned by the identity provider.
This is configurable via `RoleClaim`, `AdminRoleName`, and `FullAccessRoleName` in
[App Store configuration](#app-store-server-oidc-configuration).


## App Store server OIDC configuration

The following chart describes the various configuration options which may be either set within 
the `[OIDC]` section of the server configuration `ConfigMap` TOML, or set via environment variables.

The environment variable names in the table below need to be prefixed with `H2O_CLOUD_OIDC_`:

| TOML&nbspKey /<br/> Environment variable | <span style={{minWidth: '300px', display: 'block'}} >Description</span>| Default |
| --- | --- | --- |
| ClientID <br/> `CLIENT_ID` | Confidential client ID for authenticating browser requests | (empty)
| ClientSecret <br/> `CLIENT_SECRET` | Confidential client secret for authenticating browser requests | (empty)
| CLIClientID <br/> `CLI_CLIENT_ID` | Public client ID for authenticating CLI requests | (empty)
| WaveClientID <br/> `WAVE_CLIENT_ID` | Confidential client ID used by Apps to authenticate users.  If empty, OIDC authentication is disabled for apps. | (empty)
| WaveClientSecret <br/> `WAVE_CLIENT_SECRET` | Confidential client secret used by Apps to authenticate users. If empty, OIDC authentication is disabled for apps. | (empty)
| WaveRedirectPath <br/> `WAVE_REDIRECT_PATH` | Callback address for the OIDC provider to redirect the user after app authentications | `/_auth/callback`
| RedirectURL <br/> `REDIRECT_URL` | Callback address for the OIDC provider to redirect the user to after authentication | `http://localhost:8889/oauth2/callback`
| ProviderURL <br/> `PROVIDER_URL` | URL of OIDC provider | `http://localhost:8080/auth/realms/master`
| EndSessionURL <br/> `END_SESSION_URL` | URL to redirect the user to terminate their OIDC session | `http://localhost:8080/auth/realms/master/protocol/openid-connect/logout`
| RoleClaim <br/> `ROLE_CLAIM` | Access token claim containing the user's roles | `realm_access.roles`
| AdminRoleName <br/> `ADMIN_ROLE_NAME` | Role name assigned to administrators within the OIDC provider | `admin`
| FullAccessRoleName <br/> `FULL_ACCESS_ROLE_NAME` | Role assigned to users with "full access".  If empty, all users have full access. | (empty) 
| Scopes <br/> `SCOPES` | OIDC scopes to be granted | (empty)

## CLI authentication

All users have access to use the CLI, however, what they can do depends on the category of the user. An offline Open ID Connect (OIDC) refresh token, 
generated by the user by visiting `/auth/get-token`, is used to generate access tokens to authenticate each request from the CLI to the
App Store server.  The CLI stores the refresh token, in addition to its other configuration, on the user's machine at `~/.h2oai/h2o-cli-config.toml`.

To configure the CLI, users will need four pieces of information:

* The address of the server that the CLI will connect to
* The OpenID Connect (OIDC) provider URL to obtain access tokens
* The public OpenID Client ID designated for CLI access
* A refresh token, generated by the user by visiting `/auth/get-token` while logged in to the AI App Store web interface

## Visitors

Visitors, a.k.a., users without "full access", have limited permissions within the platform.
Users without full access privileges are considered to be visitors, and tags are used
to manually assign specific app versions to OIDC roles.
See the [Authorization section in the user guide](#authorization-for-visitors) for details.

Visitor functionality requires that `FullAccessRoleName` in the [App Store configuration](#app-store-server-oidc-configuration)
be properly configured. 

When creating a new tag, the administrator can specify an admin role and a visitor role to the tag.  Users assigned the OIDC
role that matches the admin role for the tag have the ability to assign that tag to specific apps. Users who are visitors but
assigned an OIDC role that matches the visitor role on the tag can see all app versions that tag is assigned to.
See [Tag section of the CLI documentation](#tags) on using the CLI to manipulate tags.

## Administrators

Users who log in with the OIDC role matching the server configured `AdminRoleName` are granted administrator access to the
AI App Store.  Administrators access to the administrator views within the App Store and the `admin` subcommand of the CLI.

Within the App Store, the "Admin Apps" view provides administrators with a list of every app imported into the App Store,
regardless of its visibility, and allows administrators to delete specific versions of apps. The "Admin Instances" view provides
administrators with a list of every running instance known to H2O AI Hybrid Cloud, and allows administrators to terminate specific instances of apps.

The `h2o admin` command in the CLI provides administrators with several commands, such as:

* `h2o admin app <get|list|delete>` - Manage apps owned by all users
* `h2o admin instance <get|list|status|terminate>` - Manage and inspect running app instances for all users
* `h2o admin secret <create|delete|get|list|update>` - Manage application secrets including global secrets
* `h2o admin tag <assign|create|delete|get|list|remove|update>` - Manage application tags
* `h2o admin iam <delete|export|get|import|list>` - Manage IAM policies

## Identity and Access Management (IAM) policies

**Note**: IAM policies are currently in beta and aren't enabled by default.

Exceptions to normal authorization rules can be granted by leveraging IAM policies, which can be authored in JSON and imported or exported from the platform by using the `admin iam` CLI subcommand.

IAM policies apply to one or more resources and contain one or more statements. The policy resource limits the scope of
the statements being evaluated. Statements may have the effect of either allowing or denying the request. If any one statement in any policy denies the action, then the entire request is denied.

The following is a sample IAM policy that grants users with the OIDC role `SUPPORT` the ability to list and view logs for all instances regardless of instance owner and visibility setting:

```json
{
  "resources": ["*"],
  "statements": [
    {
      "subjects": ["role:SUPPORT"],
      "actions": ["instance:view_log", "instance:read"],
      "attributes": [],
      "effect": "allow"
    }
  ]
}
```

## Wave application instance access controls

The App Store server enforces access restrictions on which users can access running applications.
See the [Authorization section in the user guide](#authorization) for details.

## Wave application user authentication

Wave applications can run as both single user instances and as multi-user instances.  Multi-user instances requires users to
first authenticate with a supported OpenID Connect (OIDC) provider.  The OIDC session created with the Wave application is
independent to the App Store user session (including the relevant OIDC client settings and tokens).
The App Store provides an opt-in mechanism to pass OpenID Connect (OIDC) endpoint, 
client, and client secret to Wave applications upon start up.  
The `WaveClientID` and `WaveClientSecret` values in the [App Store configuration](#app-store-server-oidc-configuration) 
must be set to enable OIDC integration. 
See [Wave's documentation](https://wave.h2o.ai/docs/security/#single-sign-on) for more.

\newpage

# Security

## User-Facing API security

The H2O AI Hybrid Cloud (H2OAIC) App Store relies on OpenID Connect (OIDC) to secure its API;
see the [Authorization](#authorization) for details on how the App Store leverages OIDC.

Specifically, on requests coming from the App Store web UI, it relies on OIDC-based information
stored for the browser session in the database (identified by an encrypted cookie).

For requests coming from other clients, e.g., the [CLI](#cli), it relies on
standard [Oauth2 Bearer auth](https://tools.ietf.org/html/rfc6750), where the OIDC access token
serves as a bearer token.

## System-Facing API security

Internal API calls to the API from [App instances](#app-instance), e.g., 
when downloading the [App bundle](#app-bundle-structure) for execution,
are secured using [HMAC](https://tools.ietf.org/html/rfc2104) codes identifying and authorizing 
each app instance running in the Kubernetes cluster.

## Kubernetes network security

To (optionally) secure the traffic between the App Store API and the running instances, all pod-to-pod
communication can be encrypted via a service mesh, such as  [Linkerd](http://linkerd.io).

Similarly, the App Store supports restricting the network access of the running instances and other pods
via [Kubernetes network policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/), 
e.g., to accept ingress only from the App Store server, which is responsible for all [Authorization](#authorization).

\newpage

# App configuration


## Wave server configuration

The H2O App Store lets users import and access Apps and run Instances of these apps. Primarily,
apps are written using the Python [H2O Wave](https://wave.h2o.ai) framework. H2O Wave is a web
application framework that leverages the [Wave server](https://wave.h2o.ai/docs/architecture) to
broker all interactions between the application code and a client.

In the AI App Store, each time a user runs an App, a new App Instance starts in its own pod with its
own Wave Server running along the App source code.

As a result, the AI App Store controls most of
the [Wave server configuration settings](https://wave.h2o.ai/docs/configuration#configuring-the-server).
However, an AI App Store admin can change these via environment variables, either globally for all
apps, or allow developers to change certain configuration settings on a per-app basis
via [`app.toml`](#apptoml).

This is useful, for example, when configuring maximum allowed request size via
the `H2O_WAVE_MAX_REQUEST_SIZE` setting, as the default is 5MB (requests greater than 5MB will fail
or time out), which may be too small for applications that require handling large file uploads, etc.

By default, Wave environment variables matching the following regular expressions can be changed on
per-app basis via [`app.toml`](#apptoml):

- `^H2O_WAVE_ML_.*`
- `^H2O_WAVE_PUBLIC_DIR$`
- `^H2O_WAVE_PRIVATE_DIR$`
- `^H2O_WAVE_MAX_REQUEST_SIZE$`
- `^H2O_WAVE_NO_STORE$`
- `^H2O_WAVE_SESSION_INACTIVITY_TIMEOUT$`
- `^H2O_WAVE_DATA_DIR$`
- `^H2O_WAVE_PING_INTERVAL$`

## Change a setting for all apps

As an admin, you can change a Wave setting such as a higher limit for HTTP requests for all users and
app instances in the AI App Store. 

To do this, add the required environment variable to the `apps` section in the App Store helm chart:

```yaml
apps:
  [ ... ]
  extraEnv:

    - name: H2O_WAVE_MAX_REQUEST_SIZE
      value: "25MiB"
  [ ... ]
```

**Note:**

This change is **not** applied to the already existing app instances. It will **only** be applied to
new app instances that are started after the change has been applied.



## Change a setting for a specific app

### Admin

As an admin, you can change which environment variables can be set by developers
via [`app.toml`](#apptoml).

To do this, add the required regular expression for matching allowed environment variables to
the  `config.allowedCoreEnvRegexs` section in the App Store helm chart. This is what the default
config looks like:

```yaml
config:
  [ ... ]
  allowedCoreEnvRegexs:

    - "^H2O_WAVE_ML_.*"
    - "^H2O_WAVE_PUBLIC_DIR$"
    - "^H2O_WAVE_PRIVATE_DIR$"
    - "^H2O_WAVE_NO_STORE$"
    - "^H2O_WAVE_MAX_REQUEST_SIZE$"
    - "^H2O_WAVE_SESSION_INACTIVITY_TIMEOUT$"
    - "^H2O_WAVE_DATA_DIR$"
    - "^H2O_WAVE_PING_INTERVAL$"
  [ ... ]
```

**Note:**

This change is **not** applied to the already existing apps. It will **only** be applied to new apps
that are imported after the change has been applied.



### App developer

App developers can configure the allowed environment variables via the `Env` section of
the [`app.toml`](#apptoml) file.

For example, the maximum size of HTTP requests can be changed by specifying the value
of the `H2O_WAVE_MAX_REQUEST_SIZE` variable as shown below.

```toml
[ ... ]
[[Env]]
Name = "H2O_WAVE_MAX_REQUEST_SIZE"
Value = "20M"
```


## Use a custom container image for an app

As an admin, you can set a custom container image per app version using the `h2o app set-image <appID> <image>` command. 

To do this, add the required container image regex for matching allowed environment variables to
the `config.allowedCustomImageRegexes` section in the App Store helm chart.

```yaml
config:
  [ ... ]
  allowedCustomImageRegexes:

    - "^docker.io\/h2oai\/model-manager:.*$"
  [ ... ]
```

**Note:**
You can use `".+"` to allow all custom image regexes. 


The container must set the location of the app code with the `H2O_CLOUD_APP_ROOT` environment variable and the location of the python venv
with the `H2O_CLOUD_VENV_PATH` environment variable. If the venv is created by `python -m venv /myapp/venv`, then use `H2O_CLOUD_VENV_PATH=/myapp/venv`.

\newpage

# H2O Marketplace

H2O Marketplace is a new App Store feature using which customers can obtain new apps and app
upgrades in a self-service manner directly from a central H2O-managed App Repository.

### App-scoped secrets

Unlike [Locally imported apps](#publishing-an-app-for-others-to-see-and-launch)
, each app imported via the H2O Marketplace has its own independent secret namespace that is scoped
to all versions of the app. Secrets that are scoped to an app have a visibility of `APP` and a
parent equal to the app's name
(see [Basic concepts](#app-secret) for details).

For example, if a marketplace app `sampleapp` requires a secret `test`, there has to be
secret `test` with visibility `APP` and parent `app:sampleapp`.

This restriction primarily exists to avoid unintended secret name conflicts between local apps and
marketplace apps or between two marketplace apps.

Unlike local apps, where secrets are largely managed by app developers, secrets for marketplace apps
are managed by admins; see
the [CLI documentation](#creating-and-updating-public-and-app-secrets).

\newpage

# Notifications

## Manage persistent notifications in the HAIC UI

The H2O AI Cloud Platform user interface notifies users about the success and failure of various operations using ephemeral notifications.
However, it's also possible for administrators to set persistent notifications for all users of the platform.
These persistent notifications can be used for alerting end users of a scheduled maintenance window or to remind them that
they're in a development or QA environment. Note that these notifications appear only on services using the primary UI of HAIC,
which at the time of this writing includes only the App Store. (That is, the Home, App Store, My Apps, and
My Instances views.)

When setting a persistent notification, it appears as a banner below the navigation bar in the App Store UI:


![](assets/notification_position.png)

## Set a notification in the App Store configuration

To establish a persistent notification, add the following to the `config` section in the App Store helm chart:

```yaml
config:
  [ ... ]
  extra: |
    [[Env.Notices]]
    Severity = "info"
    Title = "Attention"
    Content = "There's something you should know."
  [ ... ]
```

**Note:**

Adding multiple notifications stacks the notifications vertically in the UI. Multiple notifications will appear in the order in which they are specified.



The fields above correspond to the notification as follows:

![](assets/notification_fields.png)
The following is a list of the types of severity that you can specify:

- `info`
- `error`
- `warning`
- `blocked`
- `success`
- `severe`

Each of the preceding values produces a different icon and color for the notification:

`info`

![](assets/notification_info.png)

`error`

![](assets/notification_error.png)

`warning`

![](assets/notification_warning.png)

`blocked`

![](assets/notification_blocked.png)

`success`

![](assets/notification_success.png)

`severe`

![](assets/notification_severe.png)

\newpage

# Release notes

### 2.1.11 | Jul  22 2025

- Enhanced audit trail security and compliance: fixed permission controls to ensure audit trails are only accessible when properly enabled, strengthening your platform's security posture and compliance capabilities. 
- Fixed admin log access issues: resolved authorization problems that prevented administrators from viewing workspace logs, restoring full admin visibility and troubleshooting capabilities across all workspaces. 
- Strengthened platform security: addressed critical security vulnerabilities and updated dependencies.
- Enhanced admin interface for app instance logs: fixed admin API usage when viewing application logs, ensuring administrators can properly monitor and troubleshoot app instances regardless of workspace permissions.
- Improved admin workspace visibility: set "Show from all Workspaces" as the default view for administrators, providing immediate visibility across your entire platform without manual configuration and reducing potential confusion from scoped views. 


### 2.10.0 | Jul 15 2025

- Fixed links to help resources: Fixed Admin Center documentation links. 
- Audit trail: automatically hide Audit Trail navigation and page when the feature is unavailable, preventing user confusion and eliminating dead-end clicks that waste time. 

### 2.1.9 | Jul 14 2025

- Modify workspaces: Fixed an issue with seamlessly modifying workspace names, descriptions, and user roles after creation.
- Engine profile fields: Profile pages now show accurate field values even when some data is missing from the API response. 
- Improved user management actions: User status now properly determines available actions like password resets, making admin workflows more intuitive
- Password reset emails for active users: administrators can now send password reset emails to active users directly from the User Management page, not just pending users. 
- Set unlimited running engines: added toggle option to remove limits on maximum concurrent engines for increased flexibility 

### 2.1.8 | Jul 8 2025

- Audit trail table: allows users to view and track system events for improved transparency and troubleshooting.
- Smoother permissions management: Fixed an issue where the Update DAI Engine Profile button wouldn’t enable after assigning a custom role.
- Refine AI Engine UI: improved the UI for a more consistent and intuitive interaction experience. Added a Vsit button for Notebok engines to indicate which engines are vistable and non-visitable. 
- Git sync for Notebook Engine profiles: enables users to pull code directly into their workspace for better reproducibility and version control. 
- Discover product updates in-app: added a Beamer announcements button to the app header, making it easier for users to discover product updates from within the app.
- Organize secrets: the Secrets page now shows displayName and annotations to better organize secrets. 
- Improvements to Workspaces: The Workspace dropdown on global pages now clearly displays “Global” to avoid confusion about what data is being shown. The Workspace list is automaticaly refreshed after a deletion, saving users from manual refresh steps.

### v2.1.7 | Jun 17 2025

**UI changes**

- Improved secrets management reliability - fixed crashes on the secrets page when viewing items created or deleted by services. Service names are now clearly displayed in the details sidebar for better visibility into secret ownership. 
- Streamlined user interface - Removed visitor mode from the user menu to clean up the interface and prevent user confusion. 

### v2.1.5 | Jun 12 2025

**Release overview**

This release introduces shared Notebook Engines functionality, several UI updates to navigation, and other fixes to enhance the backend of H2O AI App Store. 

The shared Notebook Engines functionality allows teams to collaborate more effectively by enabling multiple users to access the same computational resources, while maintaining proper access controls. 

![](./notebook-engine.gif)

**UI Changes**

- Improved navigation organization: 
    - relocated the Feature Store link to the **Operations** section for better user experience and logical grouping of platform tools.
    - added Feature Store link to the main navigation menu enabling users to quickly access feature management capabilities directly from the platform interface
    - fixed navigation icon styling issues for consistent UI appearance. 
- Resolved UI display issues affecting secret management functionality: fixed truncated secret information and copy button visibility problems in the App Store.

**Backend Changes**

- Enhanced internal logging capabilities: improved support troubleshooting and faster issue resolution for customers experiencing platform issues.
- Fixed permissions issue that prevented regular users from resizing DAI Engines: ensures all authorized users can now scale their computational resources as needed. 
- Enhanced platform security: blocked unauthorized direct URL access to admin center routes, ensuring proper access controls are enforced. 

### v2.1.4 | Jun 5, 2025

**UI Changes**

- Improved navigation clarity: hidden navigation elements for uninstalled services to prevent confusion about unavailable products. 
- Enhanced Managed Cloud Admin Center experience: updated navigation naming for clarity, added cloud version display, included direct links to documentation, improved analytics with dynamic percentages, and reorganized instance management for better usability.
- Streamlined user interface interactions: eliminated unnecessary page reloads when managing secrets and aliases, fixed workspace loading delays, corrected navigation labeling, and resolved workspace picker display issues for smoother navigation

**Backend Changes**

- Prevented invalid engine upgrades: removed upgrade options from H2O Driverless AI 1.10.x to 1.11.x versions that would cause instance failures, thereby ensuring that users only see viable upgrade paths. 
- Fixed workspace-based app instance deletion: resolved issues where deleting app instances while viewing "all workspaces" would fail or behave unexpectedly.

### v2.1.3 | May 29, 2025

**UI Changes**

- Global AI Engines dashboard: administrators can now view and manage all AI engines across every workspace from a single, unified dashboard, eliminating the need to navigate between individual workspaces.

**Security Changes**

- Streamlined Admin interface: cleaned up administrative screens by removing irrelevant controls and added proper permission validation for enhanced security.

### v2.1.2 | May 19th, 2025

**UI Changes**

- Role-based navigation: improved visibility of navigation and navigational elements based on role for administrator, user, and public views.

**Security Changes**

- Streamlined public interface: cleaned up the views for a public user by removing irrelevant controls such as workspace navigation. 

**Backend Changes**

- Improved container setup: removed the requirement to set Image on every runtime and added a validation to prevent running without an Image. 

### v2.1.1 | May 14, 2025

**UI Changes**

- Admin view enhancement: full listing of all instances now available in Admin Instances view
- Improved engine details: added Engine UID field for better identification
- Better error handling: clearer error messages when navigating after workspace deletion
- User management safety: prevent accidental removal of the only workspace owner

**Security Changes**

- Authorization upgrade: added support for custom schedulers from AuthZ system
- Admin security fixes

**Backend Changes**

- Container improvements: added resources to initialization containers
- Error handling: fixed error handlers for role bindings
- Storage configuration: fixed Notebook Engine profiles by updating storage class field name

### v2.0.0 | Mar 29, 2025

**Features and improvements**

- Introduced `Workspaces` for better resource management and access control for H2O engines and applications including H2O Driverless AI. 
- Implemented the Workspace API to enhance integration capabilities. 
- Added an Admin UI for managing H2O Driverless AI and H2O Engine profiles to streamline engine configuration.  
- Enhanced H2O Driverless AI and H2O Engine and Profile configurations.
- Created Notebook Engine Profiles for better notebook performance management. 
- Enabled autopause function for specific apps to reduce resource consumption.
- Added role-based access to H2O AI Managed Cloud

**Security changes**

- Added non-root base runtime
- Added workload and resource namespaces for improved resource isolation
- Added Secure Store UI for managing secrets

**UI changes**

- Implemented a completely redesigned user interface (UI) for improved usability and modern experience.
- Added the ability to create, delete, edit and switch between Workspaces for better organization and enhanced navigation with new Workspaces dropdown to improve organization and workspace management.
- Mobile-friendly navigation: to ensure access across all devices.
- Implemented configurable logo URL to support custom branding.
- Added Workspace Switcher for H2O AI App Store to streamline application management.
- Integrated the Secure Store functionality directly into the UI.
- Implemented configuration of navigation elements for different user roles.
- Added new fields to H2O Driverless AI Engine Profile for greater customization.
- Added H2O AI App Store categories navigation for better user experience.
- Implemented ability to see upload date of apps for better version tracking.
- Added upgrade warning display for major upgrades to improve user awareness.

### v1.5.5 | Dec 19, 2024

### Core

- Updated H2O CLI to v1.0.8
- Security improvements

## v1.5.4 | Dec 18, 2024

### Core

- Updated the H2O CLI to v1.0.7.
- Updated GO Toolchain to v1.23.4.
- Removed `uuid-ossp` from Postgres.
- Added a simple RBAC authorizer for admins to control importing `H2O_CLOUD_OIDC_ADMIN_APP_IMPORTER_ROLE_NAME`. 

## v1.5.3 | Nov 29, 2024

### Core

- Added the configuration `H2O_CLOUD_ENV_NEW_MLOPS_UI_ENABLED`, which can be used to toggle whether the native H2O MLOPs UI should be present or not. 
- Added a check for the content type of file and file extensions for upload server. 

## v1.5.2 | Nov 18, 2024

### Core

- InstanceTimeout is now configurable for a specific app via the app.toml file. 

### UI

- Fixed a bug with the redirection of App Cards on the homepage. 

## v1.5.0 | Oct 21, 2024

### UI

- Added new Notebook admin settings to the UI for Kernel Images and Templates.
- Fixed a bug where Toast notifications were hiding the H2O AI Engine Manager sidebar

## v1.4.2 | Oct 09, 2024

### Core

- Added `extraIngress` and `extraEgress` to the network policy. 

## v1.4.0 | Sep 13, 2024

### Core 

- Updated golang to v1.23.1
- Added the ability to import resources from storage for H2O Managed Cloud. 
- The **ContainerOnly** mode and **Runtime image** field can be left empty for container apps. 
- All AI App Store users must now be assigned to a role. 
- The registry and version can now be templated for custom images (`{{VERSION}} {{REGISTRY}}`) by setting `config.appRegistry`. 

### UI

- Added workflow scheduling to H2O Orchestrator UI.
- Added a dashboard to H2O Orchestrator UI. 
- Adjusted the **Launch Profiles** sidebar spacing and removed borders for design consistency. 
- The instances list now gets refreshed instantly after an update.
- Introduced a mobile-friendly design for AI App Store.
- Added the DAIEngine `config.toml` to the **Details** panel of the H2O AI Engine Manager UI. 

## v1.3.0 | Aug 09, 2024

### UI

- Added access control and executor pool management to H2O Orchestrator UI.
- AI App Store now automatically logs out users that are inactive (`userInactivityTimeoutSeconds: 300`).
- Added the ability to resume a deprecated engine version to H2O AI Engine Manager UI. 

## v1.2.0 | Jul 15, 2024

### Core

- Added a new identity and access management role for allowing/disabling full access to users (`is_full_access` and `!is_full_access` roles). Full access users can now set custom images by default.
- Updated golang to v1.22.5.

### UI

- Added workspace switching to H2O Orchestrator UI.
- Fixed a bug with the **Create Alias** button.
- AI App Store UI now displays badges when an AI Engine is resizing and is about to run out of storage space.
- Users can now resize their H2O Driverless AI instances after creation.

## v1.1.1 | Jun 27, 2024

### Core

warning Deprecation notice
Deprecated Python v3.7 CPU and GPU images (this version of AI App Store is the last version to have it). 


### Core

- The AI App Store Server is now FIPS compliant. 
- Added `linux-libc-dev` back to the GPU Python v3.8 image

### UI

- Added workflow management to the H2O Orchestrator UI.

## v1.0.0 | Jun 06, 2024

### Core 

- Added a globally configurable `SHMSize` limit (default to 2Gi). 
- Added a new component of AI App Store called [H2O Orchestrator](https://docs.h2o.ai/h2o-orchestrator/) that handles orchestration and scheduling capabilities for workflows for sequential execution of scripts and notebooks. 
- Added a **Refresh Service Account Token** for telemetry reporting

### UI

- Added the ability to copy the `app.toml` code directly from the Admin Secrets UI. 

## v0.36.0 | May 03, 2024

AI App Store v0.36.0 is a patch release with some minor improvements and fixes. 

### Core

- Updated the end-user license agreement (EULA) to v04-18-2024
- The `platformUsageEnabled` variable added to helm chart. This variable allows platform admins to enable or disable the new Peak AI Unit Consumption UI page from the server config.
- Users with `Full Access` or Visitors with the correct `Visitor` roles have access to run `PUBLIC ON_DEMAND` apps. For more information about user roles and access for H2O AI Cloud, see [User roles](https://docs.h2o.ai/haic-documentation/guide/general/user-roles) in the H2O AI Cloud documentation.

### UI
Added the ability to copy `app.toml` code directly from the Admin Secrets UI. The admin can now copy the secrets from the list view of the **Admin App Secrets** page using the copy button found in the new **App.toml Code** column. The copied secret is already formatted in a way that allows admin to paste it directly onto the `app.toml` file. 

## v0.35.0 | April 15, 2024

AI App Store v0.35.0 is a patch release with some minor improvements and fixes. 

### Core

- Removed `SETUPTOOLS_USE_DISTUTILS=stdlib` for app launches in order to fix Python 3.12.
- Resolved issues when installing some Python 3.12 packages that depend on setuptools which was removed in Python 3.12, providing greater compatibility with existing Python packages.

### UI

- Administrators of AI App Store are now allowed to set default values and constraints for H2O Driverless AI and H2O Engines
- Improved `key:value` pair editing in H2O AI Engine Manager and App Secrets Manager

### CLI

- Improved error messaging when app downloading is disabled.

## v0.34.0 | March 15, 2024

AI App Store v0.34.0 is a patch release on the v0.33.0 release that makes AI Engines Settings only visible for super administrators.

- Starting and stopping AI Engines such as H2O Driverless AI, H2O Hydrogen Torch, H2O-3 via the AI Engine Manager. 
- Adjusting all configurable settings in H2O AI Cloud

For more information about the settings visible to super administrators, see [User access](#user-access). 

## v0.33.0 | March 11, 2024

Appstore v0.33.0 includes support for launch profiles for apps. This allows the server administrator to configure different profiles that can be set at launch time or set a default profile per app version. Profiles can configure `NodeSelectors`, `Tolerations`, `Affinity`, `GPU Count`, and `Memory`, `CPU Limits`, and `Reservations`.

### Core

- Added configurable launch profiles. A profile can now be customized with:
    - `Memory`, `CPU Limits`, and `Reservation`
    - `NodeSelectors`, `Affinity`, and `Tolerations`
    - `GPU Count` and `GPU Resource`
- Added the capability to set a default launch profile for a particular app. This can be done in one of two ways:
    - Run `h2o app set-default-profile` to set the default launch profile. 
    - Run `h2o app import --profile` to set the default launch profile while importing the app.  
- Added the capability to list available launch profiles by running the `h2o env launch-profiles list` command. 
- Updated goland to v.1.22
- Fixed a bug in the AI App Store CLI with using the `h2o app import --set-image` command to set the app image.

### UI

- Added a new Advanced Run app menu for selecting a profile when launching an app. 

### MISC

- Removed NPM from the Python 3.8 and Python 3.9 images
- Updated NPM in the Jupyter Python 3.10 image

## v0.31.1 & v0.31.2 | January 03, 2024

### Core

- Added `openjdk-17-jdk` back into launcher image. 

### UI

- Fixed a UI style issue for Beamer.  

## v0.31.0 | December 09, 2023

### Core

- Added support for Identity and Access Management (IAM) auth via AWS
- Upgraded Python 3.8, 3.9, and 3.10 CPU images to `debian_12 bookworm` and removed Debian version from naming
- Added new core app categories

### UI

**Improvements**

- Added filtering of app instances
- You can now transfer ownership of an AI Engine
- Added support for a [Beamer](https://www.getbeamer.com/) button. If Beamer is already configured on the environment, a newspaper icon will appear next to your username on H2O AI Cloud. Clicking it will open up the Beamer panel. 
- Renamed "Pause" to "Terminate" for H2O Engines


**Fixes**

- Fixed the Platform Usage page to use the correct units (AI units instead of mili-AI-units) and the chart to appear correctly when all data values are zero. 
- Fixed a bug in the AI Engines selector which caused inconsistent states when switching between engine options

**Note:** "Note"
The debian version prefix has been dropped from the CPU images. You can now use the aliases feature to specify the old runtime name.
```

    - name: deb_py39   
      image: 524466471676.dkr.ecr.us-east-1.amazonaws.com/q8s/launcher:latest-39
      aliases: ["deb11_py39_wlatest"]
```


## v0.30.0 | November 18, 2023

### Core

**Improvements**

- Added an additional level of visibility for apps called `PUBLIC`. Apps with `PUBLIC` visibility are visible in the app catalog and do not require authentication.
- Added an endpoint to retrieve the location or path of an app's managed instance. This lets users click on the app card on the AI App Store and directly run the instance. If no applicable instance is found, the user is redirected to the app details page. 
- Implemented RuntimeVersion Aliases
- Set empty `oidcEndSessionUrl`s to default to the server address
- Enabled downloading the CLI from the AI App Store server without accessing the internet for each specific OS

### UI

**Improvements**

- Ability to make apps `PUBLIC` mode. When an application has public mode enabled and the user does not have full access, the UI will show a streamlined version for public consumption.
- Added the ability to customize the App Store page title
- Brand theming with customized logo and brand color
- Redirect to `/logout` after OIDC logout

**Fixes**

- Fixed a bug for bulk actions in AI Engine Manager
- Fixed a bug when deleting apps using admin app UI on the admin apps page
- Fixed a bug that caused the API / CLI Page menu option to appear to users without full access when in `PUBLIC` mode


## v0.29.1 | October 30, 2023

### AI Engine Manager UI

**Improvements**

- Set the first AI Engine profile as the default
- Improved the validation message for Engine IDs

## v0.28.6. | October 17, 2023

### Core

**Improvements**

- Updated EULA
- Removed Pillow package from Jupyter Conda Environment to improve security

### UI

**Improvements**

- Added a message banner to pop up when a user has already accepted an EULA, but a new EULA is present

### MISC: Managed Cloud

**Improvements**

- Push helm chart to Managed Cloud OCI

## v0.28.4 | October 03, 2023

### Core

**Fixes**

- Fixed an issue in launcher images
- Deprecated the Python 3.7 GPU and CPU Image
- Fixed a bug when using Ephemeral Volumes in custom runtimes

## v0.28.3 | September 15, 2023

### Core

**Improvements**

- Using distroless as the base image
- The Precondition Checker is now enabled by default
- Added `H2O_WAVE_PING_INTERVAL` to the allowed wave env var list
- The Appstore WebSocket Ping Interval is now configurable 

### UI

**Improvements**

- Using Telemetry Service API URL from Discovery Service

**Fixes** 

- UI fix for scrollbars on CLI/API page
- Fixed a bug to fetch from localhost if logging service is not present

### AI Engine Manager UI

**Improvements**

- Show fallback log page if logging service is unavailable and old logs are available
- Do not allow resume for deprecated engines

## v0.28.0 | August 15, 2023

### Core

- Added a `Runtime.AppMode` container for launching docker images as apps
- The Precondition Checker API validates whether an app is runnable and scans apps periodically for validation.
- Added the `requireRuntimeVersion` config option to force apps to have a configured runtime version when importing
- Disabled the AI Unit scanning by default in preparation for the transition to telemetry service
- Added precondition checks for Apps to detect deprecated runtime versions
- Added the `PlatformUsageEnabled` config option for enabling the **Platform Usage **Page.

### UI 

**New**

- Added the **Platform Usage** Page
- Added Admin secret management

**Improvements**

- The home page is now enabled by default
- Packaged the required fonts with the UI for air-gapped environments
- Show MLOPs on CLI & API Access Page

**Fixes**

- Fixed a bug when terminating instances on the admin instance page
- Fixed a bug with displaying the proper Run button when all instances are suspended and not owned by the current user

### AI Engine Manager UI

**New**

- Updated the user flow for creating AI Engines.
- Added support for the node count field for H2O3 Engines
- Added support for searching AI Engines

**Improvements**

- Hid the **Create** button for the **Admin Engines** Page
- Updated engine size options

### CLI

**New**

- Added the `h2o admin app import cli` command to enable importing apps
- Added the `h2o app import --set-image` flag for setting a container image for both regular and admin users
- Added `h2o app list --precondition` for filtering apps based on their precondition status checks

### MISC

- Removed the `RefreshFederatedAppPreconditions` and `GetFederatedAppPreconditions` APIs
- Added Python 3.10 GPU runtime
- Added `namespaceOverride` config option for helm

## v0.27.0 | July 05, 2023

### Core

**Improvements**

- Added a config option to disable app download endpoints
- Added an admin command for redeploying an instance with the new chart and server config: `h2o admin instance redeploy`
- Made `tolerations` and `affinity` configurable per runtime version basis
- Added Instance Lifecycle to list: `h2o instance list -wide`

### UI

**Improvements** 

- Added admin UI for managing aliases
- Improvements to the CLI & API Page
- Wait for import to finish before reloading resources

### MISC

**Fixes** 

- Made a breaking change to the `RefreshFederatedAppPreconditions` and `GetFederatedAppPreconditions` federated APIs

## v0.25.3 | June 19, 2023

### Core

**Fixes**

- Fixed deadlock in `AssignAppAttributes` and `RemoveAppAttributes`.

## v0.25.2 | June 05, 2023

### Core

**Improvements**

- Added the ability to emit telemetry login events on token session exchange and limit cookie session length to access token length. 
- Allowed configuration of the `initContainerSecurityContext`.

## v0.26.0 | June 01, 2023

### Core

**New features**

- Added the ability to emit telemetry login events on token session exchange.
- Added the app name to `h2o app get` command.
- Implemented a check for custom image for federated container apps. 

**Improvements**

- Limited the cookie session length to access token length. 
- Allowed Optional Secrets for Federation.
- Allowed configuration of the `initContainerSecurityContext`.
- Allowed regular users to use alias. 

### UI

**Improvements**

- Removed Google Analytics.
- Added `requirements.txt` CodeBlock to the **CLI & API Access** page.
- Replaced "My imported apps" with "AI Engines list" widget in the home page.
- Disabled the **Visit instance** button if operation is not permitted.
- App actions no longer block the user interface. 
- Added the downloads app button to the UI.
- Allowed administrators to start managed instances from the UI.
- Removed the H2O AI Cloud version next to the About dialog title. 
- Allowed Authorization tags to show up in the **App config** table.

**Fixes**

- Fixed the AI Engine Manager API access code template.

### MISC

**Fixes**

- Removed duplicate resource request in the Helm chart.

## v0.25.1 | May 08, 2023

**Fixes**

- Fixed a bug only found in in v0.25.0 for runtime version disable packages key.

## v0.25.0 | May 02, 2023

### Core

**Improvements**

- Disabled Auto Suspension for specific instances `h2o admin instance disable-autosuspend <instanceId>`.
- Added *UnixODBC* to *ub2004_cuda114_cudnn8_py38_wlatest*.
- Disabled Init Container for Container Based Images.
- `h2o admin resume <instance>` now sets an auto suspend time.

### UI

**Improvements**

- Added non-blocking actions and error handling for **My Instances** page.
- Allowed users to copy the app name instead of the app title on **Apps List** Pages.
- Users now can easily copy the version information. 

## v0.24.1 | Apr 18, 2023

### Core

**New features**

- Users can now configure the timeout duration for bundle upload.
- Users can now configure the database connection limits.
- Screenshots can now be sorted in lexicographic order. 

**Improvements**

- Added support for the Optional Env and File secrets for Apps. 
- A config option has been added for requiring CPU and memory requests/limits in `app.toml`.

**Fixes**

- Resolved the following security vulnerabilities.
    - [CVE-2023-28840](https://github.com/advisories/GHSA-232p-vwff-86mp)
    - [CVE-2023-28841](https://github.com/advisories/GHSA-33pg-m6jh-5237)
    - [CVE-2023-28842](https://github.com/advisories/GHSA-6wrf-mxfj-pf5p)
    - [CVE-2023-28642](https://github.com/advisories/GHSA-g2j6-57v7-gm8c)
    - [CVE-2023-27561](https://github.com/advisories/GHSA-vpvm-3wq2-2wvm)
    - [CVE-2023-25809](https://github.com/advisories/GHSA-m8cg-xc2p-r3fc)

### UI

**Improvements**

- Used the Cloud Discovery to fetch OAuth configuration. 
- Sorted versions dropdown in descending order on **App details** page.
- Redesigned the **CLI & API Access Page** by including AI Engine Manager login code.

### Documentation

**Improvements**

- Restructured sections in the documentation and added more introductory content.
- Fixed documentation URL redirects. 
- Added a button to download the latest CLI to the CLI document. 

### MISC

**Improvements**

- Added Cloud Discovery service annotations for App Store.
- Added support for ingress level prefix routing for the discovery service in the Helm chart.

## v0.23.0 | Feb 28, 2023

### Core

**New features**

- Added Python CPU 3.9 and 3.10 runtimes.

**Improvements**

- Reduced overly verbose log messages.
- Reduced App Store Role permissions. 

**Fixes**

- Ensured git is present in all base images.
- Resolved the following security vulnerabilities.
    - [CVE-2023-25165](https://github.com/advisories/GHSA-pwcw-6f5g-gxf8)
    - [CVE-2022-41723](https://github.com/advisories/GHSA-vvpx-j8f3-3w6h)
    - [CVE-2023-25153](https://github.com/advisories/GHSA-259w-8hf6-59c2)
    - [CVE-2023-25173](https://github.com/advisories/GHSA-hmfx-3pcx-653p)

### UI

**Fixes**

- Fixed the back button behavior on **My Apps** and **My Instances** pages.
- Fixed the appearance of item tags in Safari.
- Fixed the **App Detail** header UI when overflowing content. 
- Fixed app tables displaying "0" when there are no tags.
- Fixed typos in the text "does not exists".
- Fixed a bug that made restarting a failed H2O Driverless AI Engine instance from the UI.  

**Improvements**

- Allowed API to send error messages directly to the UI in AI Engine Manager. 

### Documentation

**Improvements**

- Restructured `app.toml` attribute descriptions into tables to be more readable/consumable.

### MISC

**Improvements**

- Incremented packages libraries versions for the AI Notebook app.

## v0.22.1-telemetry | Jan 13, 2023

**New features**

- Added new fields for H2O MLOps telemetry.
- Added support of `affinity`, `nodeSelector`, `tolerations` and `securityContext`.
- Provided the ability to batch messages before they are sent to message queue in asynchronous server mode.

## v0.22.0 | Jan 12, 2023

### Core

**New features**

- Add support for S3 compliant object stores.
- App developers can now specify the `H2O_WAVE_SESSION_INACTIVITY_TIMEOUT` parameter in their `app.toml` file.
- Supports proxying static assets for Azure Blob Storage.
- Support for private certificates in the App Store server and apps.

**Improvements**

- Added support for Python apps and Nitro.
- Icons and screenshots are now confined to static storage only.
- The `appOwnerName` field has been added to the Kubernetes annotations, while the `appOwner` field has been added to the labels. 
- Added configurable app download timeout via `H2O_CLOUD_APP_DOWNLOAD_TIMEOUT`.
- Reduced memory usage when uploading an app via the CLI.
- Developers can now set `H2O_WAVE_SESSION_INACTIVITY_TIMEOUT` and `H2O_WAVE_DATA_DIR`.
- `H2O_CLOUD_UPGRADE_TOOLS` has been introduced to disable pip from upgrading.
- Emit a deprecation warning log message when `analyticsId` is set.
- Avoided downloading Wave server if it's available within the Python package.

**Fixes**

- Resolved a vulnerability in the `OpenPolicyAgent` dependency.

### Telemetry

**New features**

- Added support for Model Storage events and gauges.
- Added support for Model Scoring events.
- Added support for eScorer events.
- Improve scalability by adding caching to service account token reviews.
- Introduced a standalone telemetry server.
- The Tenant field is now filled on the telemetry server side.
- The telemetry server now uses the Tenant field from config.
- Provisioned telemetry service role.

### UI

**Improvements**

- My Apps v1.2 has undergone a redesign.
    - Users can now delete multiple apps simultaneously.
    - Search apps.
    - Introduced new design for "My Apps" table and "Admin Apps" table.
- Made the header consistent across all the pages.
    - Fixed active styles in navigation header links.
- Made "No data" UI consistent across all the pages.
- `EditAppPanel`: Added `app.toml` suggestion and additional metadata.
- Added search box and fixed back button behavior in **My Instances** and **Admin Instances** pages.
- AI Engine Manager:
    - Added Admin AI Engines page.
    - Added view and download logs option for AI Engines.
    - API-supplied constraints as a configuration option for H2O Driverless AI and H2O engines.
    - Allowed non-editable default values in Engine configurations. 

### Documentation

**Improvements**

- Added **Notifications** page to the **Admin guide** section, explaining on how to manage persistent notifications in the HAIC UI.

\newpage
